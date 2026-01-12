import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted


class MultitaskStackingRegressor(BaseEstimator, RegressorMixin):
    """
    True multitask stacking regressor.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base multitask regressors.

    final_estimator : estimator
        Multitask regressor trained on stacked predictions.

    n_folds : int, default=5
        Number of folds for out-of-fold prediction generation.

    shuffle : bool, default=True
        Whether to shuffle data before CV splitting.

    random_state : int or None, default=None
        Random seed for CV splitting.
    """

    def __init__(self, estimators, final_estimator, n_folds=5, shuffle=True, random_state=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 2:
            raise ValueError(
                "MultitaskStackingRegressor requires y to be 2D (n_samples, n_tasks)."
            )

        n_samples, n_tasks = y.shape
        n_estimators = len(self.estimators)

        self.n_tasks_ = n_tasks
        self.n_estimators_ = n_estimators

        # Out-of-fold meta-features
        Z = np.zeros((n_samples, n_estimators * n_tasks))

        cv = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

        self.base_estimators_ = []

        for est_idx, (name, estimator) in enumerate(self.estimators):
            oof_preds = np.zeros((n_samples, n_tasks))

            for train_idx, val_idx in cv.split(X):
                est_fold = clone(estimator)
                est_fold.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = est_fold.predict(X[val_idx])

            Z[:, est_idx * n_tasks : (est_idx + 1) * n_tasks] = oof_preds

            # Fit base estimator on full data for inference
            fitted_estimator = clone(estimator).fit(X, y)
            
            self.base_estimators_.append((name, fitted_estimator))

        # Fit multitask meta learner
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(Z, y)

        return self

    def predict(self, X):
        check_is_fitted(self, ["base_estimators_", "final_estimator_"])

        X = np.asarray(X)
        meta_features = []

        for _, est in self.base_estimators_:
            preds = est.predict(X)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            meta_features.append(preds)

        Z = np.concatenate(meta_features, axis=1)
        return self.final_estimator_.predict(Z)
