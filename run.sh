python get_data.py

cd binding
. train.sh
cd ..

cd clearance_permeability
. train.sh
cd ..

cd ksol
# . pretrain.sh
. train.sh
cd ..

cd logd
# . pretrain.sh
. train.sh
cd ..

. inference.sh
python make_submission.py
