from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import TransformedTargetRegressor


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
                RandomForestRegressor(
                    n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
                ),
            ),
        ]
    )
