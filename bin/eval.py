import argparse
import re
from pathlib import Path
from sys import path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import MODADataModule, MODADataModuleCV
from sumo.evaluation import calculate_metrics, calculate_test_metrics, plot_metrics
from sumo.model import SUMO


def custom_evaluation(datamodule, model, plot=False, ax=None):
    datamodule.prepare_data()
    overlap_thresholds = config.overlap_thresholds

    if args.test:
        precisions, recalls, f1s = calculate_test_metrics(
            model, datamodule.test_dataloader(), overlap_thresholds
        )

        if plot:
            for precision, recall, f1 in zip(precisions, recalls, f1s):
                plot_metrics(precision, recall, f1, overlap_thresholds, ax)

        return precisions, recalls, f1s
    else:
        precision, recall, f1 = calculate_metrics(
            model, datamodule.val_dataloader(), overlap_thresholds
        )

        if plot:
            plot_metrics(precision, recall, f1, overlap_thresholds, ax)

        return precision, recall, f1


def get_model(path: Union[str, Path]):
    path = Path(path)

    model_file = path if path.is_file() else get_best_model(path)
    model_checkpoint = torch.load(model_file, weights_only=False)

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


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the best UTime model of a training experiment"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to completed experiment, run with train.py",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="default",
        help="Name of configuration yaml file to use for this run",
    )
    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use test data instead of validation data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = Config(args.experiment, create_dirs=False)

    if not hasattr(config, "cross_validation") or config.cross_validation is None:
        datamodule = MODADataModule(config)
        model = get_model(args.input)

        precisions, recalls, f1s = custom_evaluation(datamodule, model, plot=True)
        precisions, recalls, f1s = (
            np.array(precisions),
            np.array(recalls),
            np.array(f1s),
        )

        if args.test:
            # results in format (dataset, overlap_threshold)
            print(np.round(f1s[:, 4], 3))
            print(np.round(f1s.mean(axis=1), 3), np.round(f1s.std(axis=1), 3))
        else:
            # results in format (overlap_threshold)
            print(np.round(f1s[4], 3))
            print(np.round(f1s.mean(), 3))
    else:
        results = []
        fold_directories = sorted(Path(args.input).glob("fold_*"))
        _, axes = plt.subplots(
            (len(fold_directories) + 2) // 3,
            3,
            figsize=(12, 8),
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()
        for fold, directory in enumerate(fold_directories):
            datamodule = MODADataModuleCV(config, fold)
            model = get_model(directory)

            results.append(
                custom_evaluation(datamodule, model, plot=True, ax=axes[fold])
            )
            axes[fold].set_title(f"Fold {fold}")
        plt.tight_layout()
        plt.show()

        if args.test:
            # results in format (dataset, metric, fold, overlap_threshold)
            results = np.array(results).transpose((2, 1, 0, 3))

            # calculate results/metrics
            # F1 at an overlap threshold of 20 percent per dataset and per fold
            f1_per_fold = results[:, 2, :, 4]
            print(
                np.round(f1_per_fold.mean(axis=1), 3),
                np.round(f1_per_fold.std(axis=1), 3),
            )

            # F1 averaged over the overlap thresholds per dataset and per fold
            f1_per_fold = results[:, 2].mean(axis=-1)
            print(
                np.round(f1_per_fold.mean(axis=1), 3),
                np.round(f1_per_fold.std(axis=1), 3),
            )
        else:
            # results in format (metric, fold, overlap_threshold)
            results = np.array(results).transpose((1, 0, 2))

            # calculate results/metrics
            # F1 at an overlap threshold of 20 percent per fold
            prec_per_fold = results[0, :, 4]
            rec_per_fold = results[1, :, 4]
            f1_per_fold = results[2, :, 4]
            print(f'Precision@20%: {np.round(prec_per_fold.mean(), 3)} ± {np.round(prec_per_fold.std(), 3)}'
                  f'\nRecall@20%: {np.round(rec_per_fold.mean(), 3)} ± {np.round(rec_per_fold.std(), 3)}'
                  f'\nF1@20%: {np.round(f1_per_fold.mean(), 3)} ± {np.round(f1_per_fold.std(), 3)}')

            # F1 averaged over the overlap thresholds per fold
            prec_per_fold = results[0].mean(axis=1)
            rec_per_fold = results[1].mean(axis=1)
            f1_per_fold = results[2].mean(axis=1)
            print(f'Mean Precision: {np.round(prec_per_fold.mean(), 3)} ± {np.round(prec_per_fold.std(), 3)}'
                  f'\nMean Recall: {np.round(rec_per_fold.mean(), 3)} ± {np.round(rec_per_fold.std(), 3)}'
                  f'\nMean F1: {np.round(f1_per_fold.mean(), 3)} ± {np.round(f1_per_fold.std(), 3)}')
