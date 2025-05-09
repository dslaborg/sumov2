# SUMOv2 - an improved version of the Slim U-Net trained on MODA

Implementation of the *SUMOv2* model as described in:

```
Grieger, N., Mehrkanoon, S., Ritter, P., and Bialonski, S., “From Sleep Staging to Spindle Detection: Evaluating End-to-End Automated Sleep Analysis”, arXiv:2505.05371, 2025.
```

## Installation Guide

On Linux and Windows, the project can be used by running the commands below to clone the repository and install the
required dependencies.

We recommend setting up a separate Python environment with `anaconda`/`miniconda`/`miniforge3` or virtualenv. In our
experiments, we used Python 3.10.
The necessary packages can then be installed via `pip`:

```shell
pip install -r requirements.txt
```

### Setup with conda

```shell
# enter the project root
git clone https://github.com/dslaborg/sumo.git
cd sumo

conda create -n sumo python=3.10
conda activate sumo
pip install -r requirements.txt
```

### Setup with virtualenv for **Linux/MacOS**

```shell
# enter the project root
git clone https://github.com/dslaborg/sumo.git
cd sumo

# Create the virtual env with pip
virtualenv venv --python=python3.10
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Setup with virtualenv for **Windows**

```shell
# enter the project root
git clone https://github.com/dslaborg/sumo.git
cd sumo

# Create the virtual env with pip
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Using the SUMO model

If you only want to use the already trained SUMO model (see `output/final.ckpt`) to detect spindles in EDF data, check
the `scripts/predict_edf_file.py` script.
On minimum, the script accepts a path to an EDF file (parameter `-ie`), and a list of channels (parameter `-c`).
Given these inputs, the script then loads the trained model, preprocesses the specified data (downsampling, passband
Butterworth filtering and normalization), predicts the spindles and writes the results including spindle frequencies and
amplitudes to an `.npz` file in `output/predict_edf_file_output`.
This `.npz` file contains one key for each channel and can be parsed as follows:

```python
import numpy as np

output_file = np.load("output/predict_edf_file_output/<file_name>.npz")
# don't be confused by the "_fold-0" appendix in the keys, the script can 
# also be used to predict spindles for a set of models in a cross validation setting
print(f"Keys in file: {list(output_file.keys())}")

for key in output_file.keys():
    print(f"Channel: {key}")
    print(f"Spindles: {output_file[key]}")
```

Optionally, you can provide the script with a path to a sleep stages file (parameter `-is`) and a different segmentation
mode (e.g., `-st block`) to only analyze spindles in N2 data.
For a full description of the parameters, see the next section below.

### predict_edf_file.py

Predict spindles in EDF data using a trained SUMOv2 model.

Arguments:

* `-e, --experiment`: name of the experiment to specify the configuration file; default is `predict`
* `-ie, --input_eeg`: path to the EDF file
* `-is, --input_sleep_stages`: path to the sleep stages file, the sleep stages should be formatted as a space separated
  list of integers corresponding to 30s epochs (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=REM); the format is the same as the
  output format of SomnoBot (https://somnobot.fh-aachen.de); optional
* `-m, --model_path`: path containing the model checkpoint, which should be used to predict spindles; default
  is `<project-dir>/output/final.ckpt`
* `-st, --seg_type`: one of `recording`, `block`, or `epoch`; specifies the segmentation mode used to split the data
  based on sleep stages: `recording` uses no segmentation, `block` splits the data into consecutive N2 blocks (requires
  sleep stages), and `epoch` splits the data into 30s epochs; default is `recording`
* `-o, --output_folder`: path in which the results should be stored; default is
  `<project-dir>/output/predict_edf_file_output`
* `-c, --channels`: space separated list of channels to use for prediction

Example call:

```bash
python scripts/predict_edf_file.py -e predict -ie "/path/to/edf/file.edf" -is "/path/to/sleep/stages/file.txt" -m "output/final.ckpt" -st block -o "output/predict_edf_file_output" -c Fp1 Fp2
```

## How to replicate the results in the paper

1. Download the [MASS data](http://ceams-carsm.ca/en/mass/) (requires application)
    1. opt: Download the [DREAMS data](https://zenodo.org/records/2650142) for evaluation. The BD dataset is not
       publicly available, but may be requested by email (see data availability statement in paper).
    2. opt: Download the [DODO/H datasets](https://github.com/Dreem-Organization/dreem-learning-open) to verify the
       sleep staging model.
2. Preprocess the MASS/MODA data using the `scripts/prepare_moda_data.py` script. Sample call:
   `python scripts/prepare_moda_data.py -p "<path_to_mass_data>/MASS/*_EDF/*PSG.edf" -o input`
3. Create data splits using the `scripts/create_data_splits.py` script. Sample call:
   `python scripts/create_data_splits.py --predefined_split "input/moda_datasplit.json" -o input`
4. Train the model using the `bin/train.py` script. Sample call: `python bin/train.py -e final`
5. Evaluate the model on the MODA test set using the `scripts/paper/eval_moda.py` script. Sample call:
   `python scripts/paper/eval_moda.py -m <model_path>`
    1. You can also do a quick check of the model on the MODA validation set using the `bin/eval.py` script. Sample
       call: `python bin/eval.py -e final -i <model_path>`
    2. opt: Evaluate the model on the DREAMS dataset using the `scripts/paper/eval_dreams.py` script. See the
       `run_evaluations.sh` script for an example call.
6. Analyze the results using the Jupyter Notebooks in `scripts/final`.

### bin/train.py

Trains the model as specified in the corresponding configuration file, writes its log to the console and saves a log
file and intermediate results for Tensorboard and model checkpoints to a result directory.

Arguments:

* `-e, --experiment`: name of experiment to run, for which a `<name>.yaml` file has to exist in the `config`
  directory; configurations inherit from the `default.yaml` configuration; default is `default`
* `-g, --config_group`: name of a configuration group that is used to group multiple runs of the same configurations in
  a single directory; useful to add structure to the output folder; optional

Example call:

```bash
python bin/train.py -e final -g replication
```

### bin/eval.py

Evaluates a trained model, either on the validation data or test data and reports the achieved metrics.

Arguments:

* `-e, --experiment`: name of configuration file, that should be used for evaluation, for which a `<name>.yaml`
  file has to exist in the `config` directory; usually equals the experiment used to train the model; default is
  `default`
* `-i, --input`: path containing the model that should be evaluated; the given input can either be a model
  checkpoint, which then will be used directly, or the output directory of a `train.py` execution, in which case the
  best model will be used from `<path>/models/`; if the configuration has cross validation enabled, the output directory
  is expected and the best model per fold will be obtained from `<path>/fold_*/models/`
* `-t, --test`: boolean flag that, if given, specifies that the test data is used instead of the validation data

Example call:

```bash
# Evaluate single model checkpoint on test data
python bin/eval.py -e final -i output/final.ckpt -t
# Evaluate best model checkpoint of several folds on validation data
python bin/eval.py -e default -i output/replication/default_<datetime>
```

### scripts/prepare_moda_data.py

Adaptation of the MODA preprocessing steps implemented in the
original [Matlab repository](https://github.com/klacourse/MODA_GC) to Python.
Preprocessing includes the following steps:

* Read data from EDF files and create required channel montages
* Filter data using a 20th order Butterworth filter between 0.3 and 30 Hz
* Downsample data to 100 Hz
* Split data into annotated 115s "blocks"
* Write data to two `.mat` files (one for each phase (young/old)) as a single vector with NaN values delimiting the
  blocks

Arguments:

* `-p, --path`: [Glob expression](https://docs.python.org/3/library/glob.html) to find all MASS `*PSG.edf` files
* `-o, --output`: path in which the generated data should be stored in; default is `<project-dir>/input/`

Example call:

```bash
python scripts/prepare_moda_data.py -p "<path_to_mass_data>/MASS/*_EDF/*PSG.edf" -o input
```

### scripts/create_data_splits.py

Demonstrates the procedure used to split the data into test and non-test subjects and the subsequent creation of a
hold-out validation set and (*alternatively*) cross validation folds.
The script allows the creation of both a new datasplit following the procedure described in the paper, or the loading of
a predefined datasplit, such as the one used in the paper.

Arguments:

* `-i, --input`: path containing the (necessary) input data, as produced by `prepare_moda_data.py`; default is
  `<project-dir>/input/`
* `--predefined_split`: optional path to a json file describing a datasplit; if given, the described datasplit is
  created instead of a new random one; default is `<project-dir>/input/moda_datasplit.json`
* `-o, --output`: path in which the generated data splits should be stored in; default is
  `<project-dir>/output/datasets_{datatime}`
* `-n, --n_datasets`: number of random split-candidates drawn/generated; default is `25`
* `-t, --test`: Proportion of data (`0<=FRACTION<=1`) that is used as test data; default is `0.2`

Example call:

```bash
# Create a new random datasplit
python scripts/create_data_splits.py --predefined_split "" -n 25 -t 0.2
# Load predefined datasplit
python scripts/create_data_splits.py --predefined_split "input/moda_datasplit.json"
```

### scripts/paper/eval_<dataset>.py

Evaluation scripts for the different datasets evaluated in the paper. Each script loads a trained model checkpoint,
sleep stages, and the EEG data. It then splits the EEG data into continuous N2 blocks, preprocesses them (steps depend
on the dataset), and evaluates the model on them. The detected spindles are finally postprocessed and spindle
characteristics are calculated.

Arguments (not all scripts require all of them):

* `-e, --experiment`: name of the experiment to specify the configuration file; default is `final`
* `-ie, --input_eeg`: path to the EDF file
* `-is, --input_sleep_stages`: path to the sleep stages file, the format depends on the dataset
* `-m, --model_path`: path containing the model checkpoint, which should be used to predict spindles
* `-st, --seg_type`: one of `recording`, `block`, or `epoch`; specifies the segmentation mode used to split the data
  based on sleep stages: `recording` uses no segmentation, `block` splits the data into consecutive N2 blocks, and
  `epoch` splits the data into 30s epochs; default is `block`
* `-o, --output_folder`: path in which the results should be stored; default is
  `<dir(model_path)>/<dataset>`
* `-c, --channels`: space separated list of channels to use for prediction

Example call:

```bash
# Excerpt 1 of the DREAMS dataset
model_path=output/final.ckpt
ie_file_name=excerpt1
ie_path=~/data/dreams/DatabaseSpindles/${ie_file_name}.edf
is_path=~/data/dreams/DatabaseSpindles/Hypnogram_${ie_file_name}.txt
python scripts/paper/eval_dreams.py -ie "$ie_path" -is "$is_path" -m $model_path -c C3-A1

# MODA dataset (uses the test split in input/subjects_val.pickle)
python scripts/paper/eval_moda.py -m $model_path
```

## Project Setup

The project is set up as follows:

* `bin/`: contains the `train.py` and `eval.py` scripts, which are used for model training and subsequent evaluation in
  experiments (as configured within the `config` directory) using
  the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework
* `config/`: contains the configurations of the experiments, configuring how to train or evaluate the model
    * `default.yaml`: provides a sensible default configuration
    * `final.yaml`: contains the configuration used to train the final model checkpoint (`output/final.ckpt`)
    * `predict.yaml`: configuration that can be used to predict spindles on arbitrary data, e.g. by using the script at
      `scripts/predict_edf_file.py`
* `input/`: contains the datasplit used in the paper (`moda_datasplit.json`); should be filled with the output of
  `prepare_moda_data.py` and the
  output of `create_data_splits.py`
* `output/`: contains generated output by any experiment runs or scripts
    * `annotations/`: contains the annotations of the DREAMS and MODA test sets used in the paper
    * `final.ckpt`: the final model checkpoint, on which the test data performance, as reported in the paper, was
      obtained; `final.log` contains the training log of the model
* `scripts/`: various scripts used to create the plots of our paper and to demonstrate the usage of this project
    * `a7/`: python implementation of the A7 algorithm (needed for datasplit) as described in:
      ```
      Karine Lacourse, Jacques Delfrate, Julien Beaudry, Paul E. Peppard and Simon C. Warby. "A sleep spindle detection algorithm that emulates human expert spindle scoring." Journal of Neuroscience Methods 316 (2019): 3-11.
      ```
    * `final/`: contains the notebooks used to create the plots of our paper
    * `moda/`: contains the necessary files from the [MODA repository](https://github.com/klacourse/MODA_GC) for
      preparing the MODA data
    * `paper/`: contains evaluation scripts for the different datasets evaluated in the paper
    * `create_data_splits.py`: demonstrates the procedure, how the data set splits were obtained, including the
      evaluation on the A7 algorithm
    * `predict_edf_file.py`: demonstrates the prediction of spindles on an arbitrary EDF file, using a trained model
      checkpoint
    * `prepare_moda_data.py`: demonstrates how to prepare the MODA data for use in this project; adapted
      from [the official MODA repository](https://github.com/klacourse/MODA_GC)
* `sumo/`: the implementation of the SUMO model and used classes and functions, for more information see the docstrings

## Configuration Parameters

The configuration of an experiment is implemented using yaml configuration files.
These files must be placed within the `config` directory and must match the name passed with `--experiment` to the
`eval.py` or `train.py` script.
The `default.yaml` is always loaded as a set of default configuration parameters and parameters specified in an
additional file overwrite the default values.
Any parameters or groups of parameters that should be `None`, have to be configured as either `null` or `Null` following
the YAML definition.

The available parameters are as follows:

* `data`: configuration of the used input data; optional, can be `None` if spindle should be annotated on arbitrary EEG
  data
    * `directory` and `file_name`: the input file containing the `Subject` objects (see `scripts/create_data_splits.py`)
      is expected to be located at `${directory}/${file_name}`; the file should be a (pickled) dict with the name of a
      data set as key and the list of corresponding subjects as value; default is `input/subjects.pickle`
    * `split`: describing the keys of the data sets to be used, specifying either `train` and `validation`, or
      `cross_validation`, and optionally `test`
        * `cross_validation`: can be either an integer k>=2, in which the keys `fold_0`, ..., `fold_{k-1}` are expected
          to exist, or a list of keys
    * `batch_size`: size of the used minbatches during training; default is `12`
    * `preprocessing`: if a RobustScaler should normalize the EEG data, default is `True`
    * `augmentation`: whether data augmentation should be applied, default is `True`
    * `augmentation_params`: parameters of the data augmentation; default is `{'scaling_factor_range': 0.5}`
* `experiment`: definition of the performed experiment; mandatory
    * `model`: definition of the model configuration; mandatory
        * `n_classes`: number of output parameters; default is `2`
        * `activation`: name of an activation function as defined in `torch.nn` package; default is `ReLU`
        * `depth`: number of layers of the U excluding the *last* layer; default is `2`
        * `channel_size`: number of filters of the convolutions in the *first* layer; default is `16`
        * `pools`: list containing the size of pooling and upsampling operations; has to contain as many values as the
          value of `depth`; default `[4;4]`
        * `convolution_params`: parameters used by the Conv1d modules
        * `moving_avg_size`: width of the moving average filter; default is `42`
    * `train`: configuration used in training the model; mandatory
        * `n_epochs`: maximal number of epochs to be run before stopping training; default is `800`
        * `early_stopping`: number of epochs without any improvement in the `val_f1_mean` metric, after which training
          is stopped; default is `300`
        * `optimizer`: configuration of an optimizer as defined in `torch.optim` package; contains `class_name` (default
          is `Adam`) and parameters, which are passed to the constructor of the used optimizer class
        * `lr_scheduler`: used learning rate scheduler; optional, default is `None`
        * `loss`: configuration of loss function as defined either in `sumo.loss` package (`GeneralizedDiceLoss`) or
          `torch.nn` package; contains `class_name` (default is `GeneralizedDiceLoss`) and parameters, which are passed
          to the constructor of the used loss class
    * `validation`: configuration used in evaluating the model; mandatory
        * `overlap_threshold_step`: step size of the overlap thresholds used to calculate (validation) F1 scores
