import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted


import torch
import torch.nn as nn
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset, DataLoader


class _CentralScrutinizerModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def nan_mse(y_hat, y):
        mask = ~torch.isnan(y)
        y_safe = torch.nan_to_num(y, nan=0.0)
        sq_error = (y_hat - y_safe) ** 2
        sq_error = sq_error * mask.float()
        column_sum_error = sq_error.sum(dim=0)
        column_valid_counts = mask.sum(dim=0).float()
        safe_counts = torch.where(column_valid_counts > 0, 
                                column_valid_counts, 
                                torch.ones_like(column_valid_counts))
        column_mse = column_sum_error / safe_counts
        total_valid_columns = (column_valid_counts > 0).float().sum()
        if total_valid_columns == 0:
            return torch.tensor(0.0, device=y.device, requires_grad=True)
            
        return column_mse.sum() / total_valid_columns

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.nan_mse(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class CentralScrutinizer(BaseEstimator, RegressorMixin):
    """
    Lightning-backed, sklearn-compatible multitask meta-regressor.
    """

    def __init__(
        self,
        hidden_dim=16,
        lr=1e-3,
        weight_decay=1e-3,
        max_epochs=200,
        batch_size=128,
        random_state=None,
        device="auto",
        verbose=True,
        output_dir=None,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.output_dir = output_dir if output_dir is not None else "central_scrutinizer"

    def fit(self, X, y):
        print(X, X.shape, y, y.shape)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim != 2:
            raise ValueError("CentralScrutinizer requires 2D y")

        n_samples, input_dim = X.shape
        _, n_tasks = y.shape

        self.input_dim_ = input_dim
        self.n_tasks_ = n_tasks

        if self.random_state is not None:
            seed_everything(self.random_state)

        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        logger = TensorBoardLogger(
            save_dir=self.output_dir,
            name="metamodel",
            default_hp_metric=False,
        )

        self.model_ = _CentralScrutinizerModule(
            input_dim=input_dim,
            n_tasks=n_tasks,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            enable_progress_bar=self.verbose,
            enable_checkpointing=False,
        )

        trainer.fit(self.model_, loader)

        self.trainer_ = trainer
        return self

    def predict(self, X):
        check_is_fitted(self, ["model_", "input_dim_", "n_tasks_"])

        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X)

        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_t).numpy(force=True)

        return preds



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
