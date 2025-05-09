import argparse
import glob
import os

import mne
import numpy as np
import scipy.io as sio
from scipy.signal import sosfiltfilt, butter, resample_poly

parent_dir = os.path.dirname(os.path.abspath(__file__))


def args():
    parser = argparse.ArgumentParser(
        description="Extract segments of EEG data from MASS dataset"
        "according to MODA annotations and save them to file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--path", type=str, help="Glob expression to find all PSG.edf files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory to save the extracted data",
        default="input",
    )
    return parser.parse_args()


def custom_parse(input_path):
    channel_file = os.path.join(parent_dir, "moda/8_MODA_primChan_180sjt.txt")
    block_files = ["moda/6_segListSrcDataLoc_p1.txt", "moda/7_segListSrcDataLoc_p2.txt"]
    eeg_files = glob.glob(input_path)
    block_length = 115

    with open(channel_file, "r") as f:
        subject_to_channel = {
            l.split("\t")[0].split(".")[0]: l.split("\t")[1][:-1]
            for l in f.readlines()[1:]
        }

    subject_to_block_start = {}
    for block_file in block_files:
        with open(os.path.join(parent_dir, block_file), "r") as f:
            for l in f.readlines()[1:]:
                l = l.strip()
                subject = l.split("\t")[1]
                block_num = int(l.split("\t")[3])
                if subject not in subject_to_block_start:
                    subject_to_block_start[subject] = {}
                subject_to_block_start[subject][block_num] = float(
                    l.split("\t")[-1].strip()
                )

    subject_to_eeg_file = {}
    for eeg_file in eeg_files:
        subject = eeg_file.split("/")[-1].split(" ")[0]
        subject_to_eeg_file[subject] = eeg_file

    subject_to_eeg_data = {}
    for subj in subject_to_block_start:
        if subj not in subject_to_eeg_file:
            print(f"WARNING: No EEG file found for subject {subj}")
            continue
        eeg_file = subject_to_eeg_file[subj]
        channel = subject_to_channel[subj]
        block_starts = subject_to_block_start[subj]
        print(f"Processing subject {subj} with channel {channel}")

        eeg_raw = mne.io.read_raw_edf(eeg_file)
        av_channels = eeg_raw.ch_names

        prim_channel = channel.split("-")[0]
        ref_channel = channel.split("-")[1]
        prim_channel = [ch for ch in av_channels if f"EEG {prim_channel}" in ch][0]
        ref_channel = [ch for ch in av_channels if f"EEG {ref_channel}" in ch]
        ref_channel = ref_channel[0] if len(ref_channel) > 0 else None
        channels_to_load = [prim_channel]
        if ref_channel is not None:
            channels_to_load.append(ref_channel)

        eeg_raw = mne.io.read_raw_edf(eeg_file, include=channels_to_load)
        eeg_data = eeg_raw.get_data()
        sr = eeg_raw.info["sfreq"]

        # re-reference channels if necessary
        if ref_channel is not None:
            channel_data = (
                eeg_data[eeg_raw.ch_names.index(prim_channel)]
                - eeg_data[eeg_raw.ch_names.index(ref_channel)]
            )
        else:
            channel_data = eeg_data[eeg_raw.ch_names.index(prim_channel)]

        # scale from V to uV
        channel_data *= 1e6

        # preprocessing
        # filter data between 0.3 and 30 Hz with a 10th order Butterworth filter
        highpass_sos = butter(10, 0.3, btype="high", fs=sr, output="sos")
        lowpass_sos = butter(10, 30, btype="low", fs=sr, output="sos")
        channel_data = sosfiltfilt(highpass_sos, channel_data)
        channel_data = sosfiltfilt(lowpass_sos, channel_data)
        # downsample to 100 Hz
        channel_data = resample_poly(channel_data, 100, int(sr))

        subject_to_eeg_data[subj] = {}
        for block_num, block_start in block_starts.items():
            block_start_idx = np.round(block_start * 100).astype(int)
            block_end_idx = block_start_idx + block_length * 100
            subject_to_eeg_data[subj][block_num] = channel_data[
                block_start_idx:block_end_idx
            ]

    return subject_to_eeg_data


if __name__ == "__main__":
    args = args()
    data = custom_parse(args.path)

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # write data to file
    # format: single column with all blocks concatenated, blocks separated by NaN
    block_files = ["moda/6_segListSrcDataLoc_p1.txt", "moda/7_segListSrcDataLoc_p2.txt"]
    n_segs_per_block = [405, 345]
    for i, block_file in enumerate(block_files):
        n_sample_in_seg = 115 * 100 + 1
        data_to_save = np.full((n_segs_per_block[i] * n_sample_in_seg, 1), np.nan)
        with open(os.path.join(parent_dir, block_file), "r") as f:
            for l in f.readlines()[1:]:
                l = l.strip()
                epoch_num = int(l.split("\t")[0])
                subject = l.split("\t")[1]
                block_num = int(l.split("\t")[3])
                if subject not in data or block_num not in data[subject]:
                    print(
                        f"WARNING: No data found for subject {subject} block {block_num}"
                    )
                    continue
                block_data = data[subject][block_num].flatten()
                seg_idx = int((epoch_num - 1) / 5 + 1)
                data_to_save[
                    (seg_idx - 1) * n_sample_in_seg : seg_idx * n_sample_in_seg - 1, :
                ] = block_data[:, np.newaxis]

        sio.savemat(
            os.path.join(output_folder, f"EEGVect_p{i + 1}.mat"),
            {"EEGvector": data_to_save},
        )
        print(f"Saved {sum(np.isnan(data_to_save)) + 1} NaNs to block {i}")
