COMMON_TEST_ARGS="
    --test-path expansion_data_test_blinded.csv \
    --smiles-columns SMILES \
    --batch-size 1024
"

chemprop predict \
    --output logd_pred.csv \
    --model-paths logd/output_train \
    $COMMON_TEST_ARGS

chemprop predict \
    --output ksol_pred.csv \
    --model-paths ksol/output_train \
    $COMMON_TEST_ARGS

chemprop predict \
    --output binding_pred.csv \
    --model-paths binding/output_train \
	--molecule-featurizers morgan_count rdkit_2d \
    $COMMON_TEST_ARGS

chemprop predict \
    --output clearance_permeability_pred.csv \
    --model-paths clearance_permeability/output_train \
    --molecule-featurizers morgan_count rdkit_2d \
    $COMMON_TEST_ARGS
