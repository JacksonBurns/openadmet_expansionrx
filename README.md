The goal is to make a stacked multi-task model, where minimol, CheMeleon, and a physicochemical random forest all simultaneously predict all outputs and then a final random forest generates the actual prediction based on the three sub-models predictions.

minimol and CheMeleon benefit from training in multi-task mode, which scikit-learn's `StackingRegressor` does not allow (because models like RandomForest do not improve with multitask, as it just fits _num_tasks_ sub-models).
Therefore, we will re-implement this ourselves (with help from AI) in `multitask_stacking_regressor.py`.

TODO: add more thorough usage and reproducibility docs

version 2:
MA-RAE: 0.60 +/- 0.02
v3: 0.60 +/- 0.03
v4: 0.59 +/- 0.02

MAE for targets (best on leaderboard right now):
LogD (0.26):         0.42 -> 0.42 -> 0.40
KSOL (0.30):         0.39 -> 0.37 -> 0.37
MLM CLint (0.31):    0.36 -> 0.37 -> 0.37 
HLM CLint (0.27):    0.30 -> 0.30 -> 0.30
Caco2 Efflux (0.27): 0.30 -> 0.33 -> 0.31
Caco2 A>B (0.19):    0.23 -> 0.26 -> 0.24
MPPB (0.14):         0.18 -> 0.16 -> 0.17
MBPB (0.11):         0.15 -> 0.13 -> 0.13
MGMB (0.14):         0.17 -> 0.17 -> 0.16
