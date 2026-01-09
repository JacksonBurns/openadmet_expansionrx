Idea:
 - for LogD and Ksol, where there is a lot of public data _and_ challenge data, pretrain models and then fine tune on challenge data
 - for other targets, train multitask models on groups of targets where they are roughly the same number of training samples (that way they are roughly evenly weighted) and the tasks make (some amount of) sense to group together, and use additional features generators inspired by molPipeline physicochemical random forest to try and improve predictions

All trained models will use 10 replicates to account for variability in validation set selection.

Final results are simply aggregated by running inference with all of the trained models.
