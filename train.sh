#!/bin/bash -x 

#dataset3

python trainDNN.py -s ../datasets/vbs_data5/ewk_train3.npy -sv ../datasets/vbs_data5/ewk_val3.npy \
        -b ../datasets/vbs_data5/wjets_train3.npy  -bv ../datasets/vbs_data5/wjets_val3.npy  \
        -ms "binary_dropout_small"  -e 200 -lr 0.005 -dr 0.05 -o ../models/run9/v5 -bs 1024 \
         -vs 70000 -p 0.00001:10


python trainDNN.py -s ../datasets/vbs_data5/ewk_train3.npy -sv ../datasets/vbs_data5/ewk_val3.npy \
        -b ../datasets/vbs_data5/wjets_train3.npy  -bv ../datasets/vbs_data5/wjets_val3.npy  \
        -ms "binary_dropout2"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v6 -bs 1024 \
         -vs 70000 -p 0.00001:10

python trainDNN.py -s ../datasets/vbs_data5/ewk_train3.npy -sv ../datasets/vbs_data5/ewk_val3.npy \
        -b ../datasets/vbs_data5/wjets_train3.npy  -bv ../datasets/vbs_data5/wjets_val3.npy  \
        -ms "binary_dropout4"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v7 -bs 1024 \
         -vs 70000 -p 0.00001:10


python trainDNN.py -s ../datasets/vbs_data5/ewk_train3.npy -sv ../datasets/vbs_data5/ewk_val3.npy \
        -b ../datasets/vbs_data5/wjets_train3.npy  -bv ../datasets/vbs_data5/wjets_val3.npy  \
        -ms "binary_dropout5"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v8 -bs 1024 \
         -vs 70000 -p 0.00001:10

# dataset 2

python trainDNN.py -s ../datasets/vbs_data5/ewk_train2.npy -sv ../datasets/vbs_data5/ewk_val2.npy \
        -b ../datasets/vbs_data5/wjets_train2.npy -bv ../datasets/vbs_data5/wjets_val2.npy  \
        -ms "binary_dropout4"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v9 -bs 256 \
         -vs 20000 -p 0.00001:10


python trainDNN.py -s ../datasets/vbs_data5/ewk_train2.npy -sv ../datasets/vbs_data5/ewk_val2.npy \
        -b ../datasets/vbs_data5/wjets_train2.npy -bv ../datasets/vbs_data5/wjets_val2.npy  \
        -ms "binary_dropout5"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v10 -bs 256 \
         -vs 20000 -p 0.00001:10

# dataset 1

python trainDNN.py -s ../datasets/vbs_data5/ewk_train.npy -sv ../datasets/vbs_data5/ewk_val.npy \
        -b ../datasets/vbs_data5/wjets_train.npy -bv ../datasets/vbs_data5/wjets_val.npy  \
        -ms "binary_dropout4"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v11 -bs 1024 \
         -vs 100000 -p 0.00001:10


python trainDNN.py -s ../datasets/vbs_data5/ewk_train.npy -sv ../datasets/vbs_data5/ewk_val.npy \
        -b ../datasets/vbs_data5/wjets_train.npy -bv ../datasets/vbs_data5/wjets_val.npy  \
        -ms "binary_dropout5"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v12 -bs 1024 \
         -vs 100000 -p 0.00001:10


python trainDNN.py -s ../datasets/vbs_data5/ewk_train.npy -sv ../datasets/vbs_data5/ewk_val.npy \
        -b ../datasets/vbs_data5/wjets_train.npy -bv ../datasets/vbs_data5/wjets_val.npy  \
        -ms "very_small_test"  -e 200 -lr 0.001 -dr 0.01 -o ../models/run9/v14 -bs 1024 \
         -vs 100000 -p 0.00001:10