import os.path
from pathlib import Path
from sys import path

import mne
import numpy as np
import pytorch_lightning as pl
import torch
from scipy import signal
from scipy.signal import butter, sosfiltfilt, resample_poly, iirfilter, detrend
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[2]))
from sumo.config import Config  # noqa: E402
from sumo.data import spindle_vect_to_indices  # noqa: E402
from sumo.model import SUMO  # noqa: E402

channels_to_aggregate = ["EEG F3-A2", "EEG F4-A1", "EEG C3-A2", "EEG C4-A1"]


def load_edf(input_file: str = None):
    input_file = Path(input_file)
    raw = mne.io.read_raw_edf(input_fname=input_file)
    eeg = dict(zip(raw.ch_names, raw.get_data()))
    return eeg, int(raw.info["sfreq"])


def load_sleep_stages(path):
    with open(path, "r") as f:
        pred_stages = f.read().split(" ")
    pred_stages = [int(s) for s in pred_stages if s != ""]
    pred_stages = [
        s if s != 5 else 4 for s in pred_stages
    ]  # map 5 (REM) to 4 (still REM)
    return pred_stages


def divide_eeg_into_segments(eeg_dict, sleep_stages, sr, seg_type):
    if seg_type == "recording":
        blocks_dict = {c: [eeg_dict[c]] for c in eeg_dict.keys()}
        starts = [0]
    elif seg_type == "block":
        sleepstages = np.repeat(
            sleep_stages, 2
        )  # transform sleepstages to half epoches
        block_mask = sleepstages == 2  # only N2 stages
        block_borders = np.diff(np.r_[0, block_mask, 0])

        # split eeg into blocks
        blocks_dict = {c: [] for c in eeg_dict.keys()}
        starts = np.argwhere(block_borders == 1) * 15
        for start, end in zip(
            np.argwhere(block_borders == 1), np.argwhere(block_borders == -1)
        ):
            start = int(start * 15 * sr)
            end = int(end * 15 * sr)
            for c in eeg_dict.keys():
                blocks_dict[c].append(eeg_dict[c][start:end])
    elif seg_type == "epoch":
        blocks_dict = {
            c: np.split(c_data, np.arange(30 * sr, len(c_data), 30 * sr))
            for c, c_data in eeg_dict.items()
        }
        starts = np.arange(len(list(blocks_dict.values())[0])) * 30
    else:
        raise ValueError("Invalid type")

    return blocks_dict, starts


def robustscale(data):
    robust_scaler = RobustScaler()
    clamp_value = 20
    x = robust_scaler.fit_transform(data[:, None])
    x[x < -clamp_value] = -clamp_value
    x[x > clamp_value] = clamp_value
    return x.flatten()


def preprocess(data, sample_rate, resample_rate):
    # filter data between 0.3 and 30 Hz with a 10th order Butterworth filter
    highpass_sos = butter(10, 0.3, btype="high", fs=sample_rate, output="sos")
    lowpass_sos = butter(10, 30, btype="low", fs=sample_rate, output="sos")
    data = [sosfiltfilt(highpass_sos, d) for d in data]
    data = [sosfiltfilt(lowpass_sos, d) for d in data]
    # downsample to 100 Hz
    data = [resample_poly(d, resample_rate, int(sample_rate)) for d in data]
    # normalize data
    data = [robustscale(d) for d in data]
    return data


def get_eegs(
    eeg: dict,
    sr: int,
    path_to_sleep_stages: str,
    seg_type: str,
    resample_rate: int = 100,
):
    channels = list(eeg.keys())
    channels = [c for c in channels if "EEG" in c]
    data = {c: eeg[c] for c in channels}

    sleep_stages = load_sleep_stages(path_to_sleep_stages)
    blocks, block_starts = divide_eeg_into_segments(data, sleep_stages, sr, seg_type)
    block_values = []
    channel_values = []
    block_start_values = []
    for c in channels:
        block_values.extend(blocks[c])
        channel_values.extend([c] * len(blocks[c]))
        block_start_values.extend(block_starts)

    eegs = preprocess(block_values, sr, resample_rate)

    return eegs, channel_values, block_start_values


def get_model(path: str, config) -> SUMO:
    model_file = Path(path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    print(f"Loading model from {model_file}")

    if torch.cuda.is_available():
        model_checkpoint = torch.load(model_file, weights_only=False)
    else:
        model_checkpoint = torch.load(
            model_file, map_location="cpu", weights_only=False
        )

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
        description="Evaluate SUMOv2 on any given eeg data"
    )
    parser.add_argument(
        "-e", "--experiment", help="experiment name (equal to config)", default="final"
    )
    parser.add_argument(
        "-ie", "--input_eeg", help="Path of the eeg file", required=True
    )
    parser.add_argument(
        "-is",
        "--input_sleep_stages",
        help="Path of the sleep stages file",
        required=True,
    )
    parser.add_argument(
        "-m", "--model_path", help="Path of the model file", required=True
    )
    parser.add_argument(
        "-st",
        "--seg_type",
        help="Type of segments to split the data into (recording, block, or epoch)",
        default="block",
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        help="Folder to save the output files",
        default="nikitin",
    )

    return parser.parse_args()


def spindle_to_onehot(spindles):
    if len(spindles) == 0:
        return np.zeros(1)
    sr = 100
    spindles_onehot = np.zeros(int(spindles[-1, 1] * sr))
    for s in spindles:
        spindles_onehot[int(s[0] * sr) : int(s[1] * sr)] = 1
    return spindles_onehot


def onehot_to_spindles(spindles_onehot):
    diff = np.diff(spindles_onehot, prepend=0, append=0)
    start = np.where(diff == 1)[0] / 100
    end = np.where(diff == -1)[0] / 100
    return np.vstack([start, end]).T


def aggregate_spindles(spindles_set):
    spindles_set_1hot = [spindle_to_onehot(spindles) for spindles in spindles_set]
    spindles_agg = np.zeros(max([len(s) for s in spindles_set_1hot]))
    for s in spindles_set_1hot:
        spindles_agg[: len(s)] += s
    spindles_agg = (spindles_agg > 0).astype(int)
    return onehot_to_spindles(spindles_agg)


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

    input_eeg = args.input_eeg
    input_f_id = input_eeg.split("/")[-1].split(".")[0]
    input_sleep_stages = args.input_sleep_stages

    resample_rate = 100

    config = Config(args.experiment, create_dirs=False)

    full_eeg, original_sr = load_edf(input_eeg)
    eegs, channels, block_starts = get_eegs(
        full_eeg,
        original_sr,
        input_sleep_stages,
        resample_rate=resample_rate,
        seg_type=args.seg_type,
    )

    dataset = SimpleDataset(eegs)
    dataloader = DataLoader(dataset)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_sanity_val_steps=0,
        logger=False,
    )

    directory = Path(args.model_path)
    analyzed_spindles = {f"{input_f_id}_{ch}_fold-0_agg": [] for ch in channels}

    model = get_model(str(directory), config)

    predictions = trainer.predict(model, dataloader)

    predictions_per_channel = {c: [] for c in channels}
    for i, (channel, x, pred, b_start) in enumerate(
        zip(channels, eegs, predictions, block_starts)
    ):
        spindle_vect = pred[0].numpy()
        spindles = spindle_vect_to_indices(spindle_vect) / resample_rate
        spindles += b_start
        predictions_per_channel[channel].append(spindles)
    predictions_per_channel = {
        c: np.concatenate(sp) if len(sp) > 0 else np.zeros((0, 2))
        for c, sp in predictions_per_channel.items()
    }

    # aggregated spindles
    spindles_to_aggregate = [
        predictions_per_channel[ch] for ch in channels_to_aggregate
    ]
    aggregated_spindles = aggregate_spindles(spindles_to_aggregate)

    # filter and merge spindles
    if len(aggregated_spindles) > 0:
        aggregated_spindles = merge_and_filter_spindles(aggregated_spindles)

    # analyze frequencies
    for i, channel in enumerate(channels_to_aggregate):
        x = full_eeg[channel]
        bandpass_frequencies = [10, 16]
        sos = iirfilter(
            2,
            [
                bandpass_frequency * 2.0 / original_sr
                for bandpass_frequency in bandpass_frequencies
            ],
            btype="bandpass",
            ftype="butter",
            output="sos",
        )
        x_filtered = sosfiltfilt(sos, x)

        for spindle in aggregated_spindles:
            start = int(spindle[0] * original_sr)
            end = int(spindle[1] * original_sr)
            spindle_eeg = x_filtered[start:end]
            spindle_eeg_unfiltered = x[start:end]

            avg_frequency_zero_cross = calculate_frequencies(spindle_eeg, original_sr)

            sp_amp_ptp, sp_amp_hilbert = calculate_amplitude(
                spindle_eeg_unfiltered, spindle_eeg
            )

            analyzed_spindles[f"{input_f_id}_{channel}_fold-0_agg"].append(
                [
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
    output_dir = Path(output_dir) / args.output_folder / args.seg_type
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        file=str(output_dir / (input_f_id + f"_aggregated.npz")),
        **analyzed_spindles,
    )


if __name__ == "__main__":
    main()
