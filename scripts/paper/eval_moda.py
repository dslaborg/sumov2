import os.path
import pickle
from pathlib import Path
from sys import path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import signal
from scipy.signal import sosfiltfilt, iirfilter, detrend
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[2]))
from sumo.config import Config  # noqa: E402
from sumo.data import spindle_vect_to_indices  # noqa: E402
from sumo.model import SUMO  # noqa: E402


def robustscale(data):
    robust_scaler = RobustScaler()
    clamp_value = 20
    x = robust_scaler.fit_transform(data[:, None])
    x[x < -clamp_value] = -clamp_value
    x[x > clamp_value] = clamp_value
    return x.flatten()


def preprocess(data):
    # files are already filtered and downsampled
    data = [robustscale(d) for d in data]
    return data


def load_eeg_data(input_file: str):
    input_file = Path(input_file)
    subjects = pickle.load(open(input_file, "rb"))["test"]
    eeg_data = {}
    phase_s_id_mapping = {}
    for subjects_phase in subjects:
        phase_key = f"test_{subjects_phase[0].phase}"
        phase_s_id_mapping[phase_key] = [s.patient_id for s in subjects_phase]
        for s in subjects_phase:
            eeg_data[s.patient_id] = [d for d in preprocess(s.data)]
    return eeg_data, phase_s_id_mapping


def get_model(path: str, config) -> SUMO:
    model_file = Path(path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    print(f"Loading model from {model_file}")

    if torch.cuda.is_available():
        model_checkpoint = torch.load(model_file, weights_only=False)
    else:
        model_checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)

    model = SUMO(config)
    model.load_state_dict(model_checkpoint["state_dict"])

    return model


class SimpleDataset(Dataset):
    def __init__(self, data_vectors):
        super(SimpleDataset, self).__init__()

        self.data = data_vectors

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a UTime model on any given eeg data"
    )
    parser.add_argument(
        "-e", "--experiment", help="experiment name (equal to config)", default="final"
    )
    parser.add_argument(
        "-m", "--model_path", help="Path of the model file", required=True
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        help="Folder to save the output files",
        default="moda",
    )

    return parser.parse_args()


def calculate_frequencies(eeg_signal, sample_rate):
    zero_crossings = np.where(np.diff(np.sign(eeg_signal)))[0]
    instantaneous_frequency = sample_rate / np.diff(zero_crossings) / 2
    avg_frequency_zero_cross = (
        np.mean(instantaneous_frequency) if len(instantaneous_frequency) > 0 else 0
    )

    return avg_frequency_zero_cross


def calculate_amplitude(eeg_signal_unfiltered, eeg_signal_filtered):
    # Important: detrend the spindle signal to avoid wrong peak-to-peak amplitude
    sp_det = detrend(eeg_signal_unfiltered, type="linear")

    # Now extract the peak to peak amplitude
    sp_amp_ptp = np.ptp(sp_det)  # Peak-to-peak amplitude

    analytic = signal.hilbert(eeg_signal_filtered)
    sp_amp_hilbert = np.mean(np.abs(analytic))

    return sp_amp_ptp, sp_amp_hilbert


def merge_and_filter_spindles(
    spindles, merge_dur=0.3, merge_dist=0.1, min_dur=0.3, max_dur=2.5
):
    sort_idx = np.argsort(spindles[:, 0])
    spindles = spindles[sort_idx]
    durations = spindles[:, 1] - spindles[:, 0]
    distances = spindles[1:, 0] - spindles[:-1, 1]
    to_merge = (
        (durations[:-1] < merge_dur)
        & (durations[1:] < merge_dur)
        & (distances < merge_dist)
    )
    spindles[np.r_[to_merge, False], 1] = spindles[np.r_[False, to_merge], 1]
    spindles = spindles[~np.r_[False, to_merge]]

    durations = spindles[:, 1] - spindles[:, 0]
    to_filter = (durations < min_dur) | (durations > max_dur)
    spindles = spindles[~to_filter]

    return spindles


def main():
    args = get_args()

    config = Config(args.experiment, create_dirs=False)
    input_eeg = "input/subjects_val.pickle"
    model = get_model(args.model_path, config)

    eeg_data, phase_s_id_mapping = load_eeg_data(input_eeg)

    phases_keys = [k for k in phase_s_id_mapping.keys() if "test" in k]
    phases_data = {k: phase_s_id_mapping[k] for k in phases_keys}

    analyzed_spindles = {}
    for phase_key, phase_s_ids in phases_data.items():
        for s_id in phase_s_ids:
            print(f"Analyzing {s_id}")
            analyzed_spindles[f"{s_id}_{phase_key}"] = []

            dataset = SimpleDataset(eeg_data[s_id])
            dataloader = DataLoader(dataset)

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                num_sanity_val_steps=0,
                logger=False,
            )

            predictions = trainer.predict(model, dataloader)

            predictions_in_sec = []
            for i, block_pred in enumerate(predictions):
                spindle_vect = block_pred[0].numpy()
                spindles = spindle_vect_to_indices(spindle_vect) / 100
                predictions_in_sec.append(spindles)

            for block_num, block_spindles in enumerate(predictions_in_sec):
                if len(block_spindles) == 0:
                    continue
                # merge and filter spindles
                block_spindles = merge_and_filter_spindles(block_spindles)

                # analyze frequencies
                x = eeg_data[s_id][block_num]
                bandpass_frequencies = [10, 16]
                sos = iirfilter(
                    2,
                    [
                        bandpass_frequency * 2.0 / 100
                        for bandpass_frequency in bandpass_frequencies
                    ],
                    btype="bandpass",
                    ftype="butter",
                    output="sos",
                )
                x_filtered = sosfiltfilt(sos, x)

                for spindle in block_spindles:
                    start = int(spindle[0] * 100)
                    end = int(spindle[1] * 100)
                    spindle_eeg = x_filtered[start:end]
                    spindle_eeg_unfiltered = x[start:end]
                    if len(spindle_eeg) == 0:
                        continue

                    avg_frequency_zero_cross = calculate_frequencies(spindle_eeg, 100)

                    sp_amp_ptp, sp_amp_hilbert = calculate_amplitude(
                        spindle_eeg_unfiltered, spindle_eeg
                    )

                    analyzed_spindles[f"{s_id}_{phase_key}"].append(
                        [
                            block_num,
                            spindle[0],
                            spindle[1],
                            avg_frequency_zero_cross,
                            sp_amp_ptp,
                            sp_amp_hilbert,
                        ]
                    )

    output_dir = (
        os.path.dirname(args.model_path)
        if os.path.isfile(args.model_path)
        else args.model_path
    )
    output_dir = Path(output_dir) / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        file=str(output_dir / f"all_aggregated.npz"),
        **analyzed_spindles,
    )


if __name__ == "__main__":
    main()
