from pathlib import Path

import numpy as np
import pandas as pd


if __name__ == "__main__":
    _df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv")
    # clip the highest 5% of values for each target to reduce the impact of extreme outliers
    for target in ["LogD","KSOL","HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux","MPPB","MBPB","MGMB"]:
        upper_bound = _df[target].quantile(0.95)
        _df.loc[_df[target] > upper_bound, target] = upper_bound

    # log transform certain targets and save out separate CSVs for each target group
    for target in [["LogD"],["KSOL"],["HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux", "MPPB","MBPB","MGMB"]]:
        if target[0] != "LogD":  # log transform
            _df[target] = np.log1p(_df[target])
        _df[["SMILES"] + target].dropna(subset=target, how="all").to_csv(f"train_{"_".join(target).replace(" ", "_").replace(">", "gt")}.csv", index=False)

    # finally save the parent
    _df.to_csv("expansion_data_train_raw.csv", index=False)

    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv")
    df.to_csv("expansion_data_test_blinded.csv", index=False)
