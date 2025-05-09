import os.path
import re
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
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config  # noqa: E402
from sumo.data import spindle_vect_to_indices  # noqa: E402
from sumo.model import SUMO  # noqa: E402


def load_edf(input_file: str, channels_to_include):
    input_file = Path(input_file)
    raw = mne.io.read_raw_edf(input_fname=input_file, include=channels_to_include)
    eeg = dict(zip(raw.ch_names, raw.get_data()))
    # eeg = {k: v * 1e6 for k, v in eeg.items()}  # Convert to microvolts
    return eeg, int(raw.info["sfreq"])


def load_sleep_stages(path):
    """
    Parsing function for sleep stage files. Currently, this parser is written for the output format of SomnoBot
    (https://somnobot.fh-aachen.de), which consists of a single line with the sleep stages separated by spaces.
    """
    with open(path, "r") as f:
        pred_stages = f.read().strip().split(" ")
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
        if sleep_stages is None:
            raise ValueError("Sleep stages are required for block segmentation")

        sleepstages = np.repeat(
            sleep_stages, 2
        )  # transform sleepstages to half epoches
        next_epoch = np.hstack(
            [sleepstages[1:], [-1]]
        )  # overlap last half of current epoch with the next epoch

        # TODO: change this filter if you want to detect spindles in N1/N2/N3 instead of just N2
        block_mask = (  # there can only be a spindle if
            sleepstages
            == 2  # a) current epoch is "N2"
            # | (sleepstages == 3)  # b) current epoch is "N3"
            # | (  # c) current epoch is "N1" and next epoch is "N2" or "N3"
            #     (sleepstages == 1) & ((next_epoch == 2) | (next_epoch == 3))
            # )
        )
        block_borders = np.diff(np.r_[0, block_mask, 0])

        # split eeg into blocks
        blocks_dict = {c: [] for c in eeg_dict.keys()}
        starts = np.argwhere(block_borders == 1) * 15
        for start, end in zip(
            np.argwhere(block_borders == 1)[:, 0],
            np.argwhere(block_borders == -1)[:, 0],
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
    lowpass = 30 if sample_rate > 60 else (sample_rate - 1) / 2
    lowpass_sos = butter(10, lowpass, btype="low", fs=sample_rate, output="sos")
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
    data = {c: eeg[c] for c in channels}

    if path_to_sleep_stages is None:
        sleep_stages = None
    else:
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
    path = Path(path)
    model_file = path if path.is_file() else get_best_model(path)
    print(f"Loading model from {model_file}")

    if torch.cuda.is_available():
        model_checkpoint = torch.load(model_file, weights_only=False)
    else:
        model_checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)

    model = SUMO(config)
    model.load_state_dict(model_checkpoint["state_dict"])

    return model


def get_best_model(experiment_path: Path, sort_by_loss: bool = False):
    models_path = experiment_path / "models"
    models = list(models_path.glob("epoch=*.ckpt"))

    regex = (
        r".*val_loss=(0\.[0-9]+).*\.ckpt"
        if sort_by_loss
        else r".*val_f1_mean=(0\.[0-9]+).*\.ckpt"
    )
    regex_results = [re.search(regex, str(m)) for m in models]

    models_score = np.array([float(r.group(1)) for r in regex_results])
    model_idx = np.argmin(models_score) if sort_by_loss else np.argmax(models_score)

    return models[model_idx]


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
        "-e", "--experiment", help="experiment name (equal to config)", default="predict"
    )
    parser.add_argument(
        "-ie", "--input_eeg", help="Path of the EDF file", required=True
    )
    parser.add_argument(
        "-is",
        "--input_sleep_stages",
        help="Path of the sleep stages file",
        required=False,
    )
    parser.add_argument(
        "-m", "--model_path", help="Path of the model file", default="output/final.ckpt"
    )
    parser.add_argument(
        "-st",
        "--seg_type",
        help="Type of segments to split the data into (recording, block, or epoch)",
        default="recording",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="Folder to save the output files",
        default="predict_edf_file_output",
    )
    parser.add_argument(
        "-c",
        "--channels",
        help="Channels to use",
        nargs="+",
        required=True,
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

    input_eeg = args.input_eeg
    input_f_id = input_eeg.split("/")[-1].split(".")[0]
    input_sleep_stages = args.input_sleep_stages

    resample_rate = 100

    config = Config(args.experiment, create_dirs=False)

    full_eeg, original_sr = load_edf(input_eeg, args.channels)
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

    # this script can also be used to evaluate the models of a cross-validation setup if the experiment file has cross
    # validation enabled (e.g., the "default" experiment file)
    if hasattr(config, "cross_validation") and config.cross_validation:
        fold_directories = sorted(Path(args.model_path).glob("fold_*"))
        analyzed_spindles = {
            f"{ch}_fold-{fold}": []
            for ch in channels
            for fold in range(len(fold_directories))
        }
    else:
        fold_directories = [Path(args.model_path)]
        analyzed_spindles = {f"{ch}_fold-0": [] for ch in channels}

    for fold, directory in enumerate(fold_directories):
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

        # analyze frequencies
        for i, channel in enumerate(predictions_per_channel.keys()):
            # filter and merge spindles
            filtered_spindles = merge_and_filter_spindles(
                predictions_per_channel[channel]
            )

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

            for spindle in filtered_spindles:
                start = int(spindle[0] * original_sr)
                end = int(spindle[1] * original_sr)
                spindle_eeg = x_filtered[start:end]
                spindle_eeg_unfiltered = x[start:end]

                avg_frequency_zero_cross = calculate_frequencies(
                    spindle_eeg, original_sr
                )

                sp_amp_ptp, sp_amp_hilbert = calculate_amplitude(
                    spindle_eeg_unfiltered, spindle_eeg
                )

                analyzed_spindles[f"{channel}_fold-{fold}"].append(
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
        file=str(output_dir / (input_f_id + f".npz")),
        **analyzed_spindles,
    )


if __name__ == "__main__":
    main()
