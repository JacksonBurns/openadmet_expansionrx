The goal is to make a stacked multi-task model, where minimol, CheMeleon, and a physicochemical random forest all simultaneously predict all outputs and then a final random forest generates the actual prediction based on the three sub-models predictions.

minimol and CheMeleon benefit from training in multi-task mode, which scikit-learn's `StackingRegressor` does not allow (because models like RandomForest do not improve with multitask, as it just fits _num_tasks_ sub-models).
Therefore, we will re-implement this ourselves (with help from AI) in `multitask_stacking_regressor.py`.
