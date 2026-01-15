# @JacksonBurns' OpenADMET x ExpansionRx Submission

This repository holds the code needed to generate my submissions to the OpenADMET x ExpansionRx machine learning competition, as well as a high-level description of the model (below).

Installation and usage instructions are also included, for those seeking to reproduce or extend these results.

## Model Description

### LogD and KSOL

For these two tasks, there is enough data that we afford to train a Chemprop model from scratch and expect good performance.
This is done in `ksol_logd`, where we optimize a single multitask model to predict both at the same time, and then train 5 replicates of it for later prediction.
Very standard stuff there.

### Binding, Permeability, and Clearance

For the remaining tasks, the available data is much more limited and different techniques are needed.

Multitask learning, where a deep learning model simultaneously predicts multiple outputs at the same time, has been shown to be highly effective.
At the same time, classical approaches are often strong even by comparison, often for reasons that are hard to detect.

To take advantage of both, we will train a Multitask Stacking Model, where `minimol` ([architecture here](./minimolregressor/model.py)), `CheMeleon` ([architecture here](./chemeleonregressor/model.py)), and a physicochemical (Morgan Count @ 2048 with RDKit descriptors) random forest all make predictions for a given input, and then a final meta-model combines them to produce the final result.

`minimol` and `CheMeleon` are easy to do this with, but scikit-learn's `StackingRegressor` does not allow it for random forest (because models like RandomForest do not improve with multitask, as it just fits _num_tasks_ sub-models).
Therefore, we will re-implement this ourselves (with help from AI) in `multitask_stacking_regressor.py` (which also houses the final meta-model).

## Installation

Two separate environments are needed - one with `minimol`, and one with everything else.

### `minimol`

`minimol` doesn't play well with other dependencies installed, so I put it into its own environment and save its outputs to disk instead.
You can see _my_ installation script [here](./minimolregressor/install.sh) - I suggest adapting this to your own needs, paying special attention to:
 - scipy: more recent versions than the one I have listed do not work
 - numpy: cannot tolerate any of the new v2 releases
 - torch + cuda: if you're using a GPU, install the version of PyG with the appropriate CUDA version and make sure the correponding torch version is reflected elsewhere.

### Everything Else

During initial development Python 3.12 was used, though 3.11 should also work.
Within a virtual environment, you should install these dependencies:

```
pandas
numpy
scikit_learn
scikit_mol
chemprop[hpopt]>=2.2.1,<2.3
```

Chemprop 2.2.1 currently has bugs with `lightning>=2.6.0` and `ray[tune]>2.45`, so you might need to limit the versions of those installed packages _or_ install this exact version of Chemprop:

`pip install "chemprop[hpopt] @ git+https://github.com/chemprop/chemprop@5f8d951b6cf3677864f81f1fd056cb4f1af75816"`

Once Chemprop 2.2.2 is out, this shouldn't be an issue anymore.

## Usage

### Setup

Run `python get_data.py` to retrieve and apply some pre-processing to the challenge data.

Run `features.py` inside `minimolregressor` to generate the `minimol` learned representations and save them to disk.

### Training

KSOL and LogD models are fit using the Chemprop CLI, so you cna just navigate to their directory and execute the shell scripts (i.e., `. train.sh`).
The results of the hyperparameter optimization (`opt.sh`) are hard-coded into `train.sh`, so you don't need to re-execute `hpopt` unless you want to.
Once the model is trained, you should run `predict.sh` to generate the predictions for LogD and KSOL.

For the remaining targets, you should run `python predict.py /path/to/output` where `/path/to/output` is where you want checkpoints, log files, and the actual model (`model.joblib`) to be deposited.

### Inference

To run predictions, execute `python predict.py /path/to/output/.../model.joblib /path/to/ksol_logd/predictions.csv` where the former argument is the path to the trained model and the latter is the predictions from the separate KSOL and LogD model.
This script will generate a `test_predictions_[...].csv` timestamped file that is suitable for submission to the competition.
The preprocessing (namely re-scaling and clipping) done in the setup stage will also be handled in this script.

## Development Notes

The below are just some notes about the earlier versions of the model, preserved for posterity.

v2: initial attempt - no replicates for the submodels
v3: 5-fold cross val for the submodels, which imprved some targets but cost on others
v4: 10 fold cross val in stacking model and add an additional 10% for training the 5 replicates of minimol and CheMeleon; seemed to be more of a 'best of both worlds'
v5: weighting by task, which had little to no effect
v6: ksol and logd separate model, NN for meta-model

Overall Stats - MA-RAE:
v2: 0.60 +/- 0.02
v3: 0.60 +/- 0.03
v4: 0.59 +/- 0.02
v5: 0.59 +/- 0.02
v6: 0.57 +/- 0.02

MAE for targets (best on leaderboard right now):
LogD (0.26):         0.42 -> 0.42 -> 0.40 -> 0.41 -> 0.33
KSOL (0.30):         0.39 -> 0.37 -> 0.37 -> 0.37 -> 0.34
MLM CLint (0.31):    0.36 -> 0.37 -> 0.37 -> 0.37 -> 0.37
HLM CLint (0.27):    0.30 -> 0.30 -> 0.30 -> 0.30 -> 0.30
Caco2 Efflux (0.27): 0.30 -> 0.33 -> 0.31 -> 0.31 -> 0.32
Caco2 A>B (0.19):    0.23 -> 0.26 -> 0.24 -> 0.24 -> 0.23
MPPB (0.14):         0.18 -> 0.16 -> 0.17 -> 0.18 -> 0.16
MBPB (0.11):         0.15 -> 0.13 -> 0.13 -> 0.13 -> 0.14
MGMB (0.14):         0.17 -> 0.17 -> 0.16 -> 0.15 -> 0.17
