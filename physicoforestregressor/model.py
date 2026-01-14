from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class ColumnwiseRFRegressor(BaseEstimator, RegressorMixin):
    """
    Fits a separate RandomForestRegressor to each column of a 2D target y.
    Rows with NaN in a target column are dropped for that column.
    """

    def __init__(self, n_estimators=100, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        n_targets = y.shape[1]
        self.models_ = []

        for col in range(n_targets):
            # Select rows where target is not nan
            mask = ~np.isnan(y[:, col])
            X_col = X[mask]
            y_col = y[mask, col]

            # Clone a new RandomForestRegressor for this column
            model = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs
            )
            model.fit(X_col, y_col)
            self.models_.append(model)

        return self

    def predict(self, X):
        X = np.asarray(X)
        # Predict each column separately and stack horizontally
        preds = [model.predict(X) for model in self.models_]
        return np.column_stack(preds)


def get_prf_pipe(
    morgan_radius: int = 2, morgan_size: int = 2048, n_estimators: int = 500, random_seed: int = 42
):
    return Pipeline(
        [
            ("smiles2mol", SmilesToMolTransformer()),
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "morgan",
                            MorganFingerprintTransformer(
                                fpSize=morgan_size, radius=morgan_radius, useCounts=True, n_jobs=-1
                            ),
                        ),
                        (
                            "physchem",
                            MolecularDescriptorTransformer(
                                desc_list=[
                                    desc
                                    for desc in MolecularDescriptorTransformer().available_descriptors
                                    if desc != "Ipc"
                                ],
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                ColumnwiseRFRegressor(
                    n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
                ),
            ),
        ]
    )
