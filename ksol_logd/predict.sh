chemprop predict \
    --output logd_ksol_pred.csv \
    --model-paths output_train \
    --test-path ../test.csv \
    --smiles-columns SMILES \
	--molecule-featurizers rdkit_2d \
    --batch-size 1024
