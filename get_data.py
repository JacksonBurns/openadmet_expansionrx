from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    data_cache_f = Path("expansion_data_train_raw.csv")
    if data_cache_f.exists():
        _df = pd.read_csv(data_cache_f)
    else:
        _df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv")
        _df.to_csv(data_cache_f, index=False)

    for target in [["LogD"],["KSOL"],["HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux"],["MPPB","MBPB","MGMB"]]:
        _df[["SMILES"] + target].dropna(subset=target).to_csv(f"train_{"_".join(target).replace(" ", "_").replace(">", "gt")}.csv", index=False)

    data_cache_f = Path("expansion_data_test_blinded.csv")
    if data_cache_f.exists():
        df = pd.read_csv(data_cache_f)
    else:
        df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv")
        df.to_csv(data_cache_f, index=False)
