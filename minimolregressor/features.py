if __name__ == "__main__":
    import pandas as pd
    from minimol import Minimol

    model = Minimol()

    dfs = []
    for subset in ["train", "test"]:
        df = pd.read_csv(f"../{subset}.csv")
        smiles = df["clean_smiles"].tolist()
        features = [t.tolist() for t in model(smiles)]
        feat_df = pd.DataFrame(
            data=features, columns=[f"feature_{i}" for i in range(512)], index=smiles
        )
        dfs.append(feat_df)
    combined_df = pd.concat(dfs, axis=0)
    combined_df.to_parquet("../minimol_features.parquet")
