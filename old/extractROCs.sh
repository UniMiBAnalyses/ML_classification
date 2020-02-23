#!/bin/bash -x 

# model t0
python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t0_val_run7.npy  \
                        -m ../models/run7/t0_v5.yaml  -o ../ROC_curves/run7/qcd_t0  -f roc_model_t0_ewk_t0

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t1_val_run7.npy  \
                        -m ../models/run7/t0_v5.yaml  -o ../ROC_curves/run7/qcd_t0 -f roc_model_t0_ewk_t1

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t2_val_run7.npy  \
                        -m ../models/run7/t0_v5.yaml  -o ../ROC_curves/run7/qcd_t0 -f roc_model_t0_ewk_t2


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/w+jets_val_run7.npy  \
                        -m ../models/run7/t0_v5.yaml  -o ../ROC_curves/run7/qcd_t0 -f roc_model_t0_ewk_w+jets


#model t1

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t0_val_run7.npy  \
                        -m ../models/run7/t1.yaml  -o ../ROC_curves/run7/qcd_t1  -f roc_model_t1_ewk_t0

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t1_val_run7.npy  \
                        -m ../models/run7/t1.yaml  -o ../ROC_curves/run7/qcd_t1  -f roc_model_t1_ewk_t1 

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t2_val_run7.npy  \
                        -m ../models/run7/t1.yaml  -o ../ROC_curves/run7/qcd_t1  -f roc_model_t1_ewk_t2


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/w+jets_val_run7.npy  \
                        -m ../models/run7/t1.yaml  -o ../ROC_curves/run7/qcd_t1  -f roc_model_t1_ewk_w+jets

#model t2


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t0_val_run7.npy  \
                        -m ../models/run7/t2.yaml  -o ../ROC_curves/run7/qcd_t2  -f roc_model_t2_ewk_t0

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t1_val_run7.npy  \
                        -m ../models/run7/t2.yaml  -o ../ROC_curves/run7/qcd_t2   -f roc_model_t2_ewk_t1 

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t2_val_run7.npy  \
                        -m ../models/run7/t2.yaml  -o ../ROC_curves/run7/qcd_t2  -f roc_model_t2_ewk_t2


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/w+jets_val_run7.npy  \
                        -m ../models/run7/t2.yaml  -o ../ROC_curves/run7/qcd_t2  -f roc_model_t2_ewk_w+jets


#model w+jets


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t0_val_run7.npy  \
                        -m ../models/run7/w+jets_v5.yaml  -o ../ROC_curves/run7/w+jets  -f roc_model_w+jets_ewk_t0

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t1_val_run7.npy  \
                        -m ../models/run7/w+jets_v5.yaml  -o ../ROC_curves/run7/w+jets  -f roc_model_w+jets_ewk_t1

python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/qcd_t2_val_run7.npy  \
                        -m ../models/run7/w+jets_v5.yaml  -o ../ROC_curves/run7/w+jets  -f roc_model_w+jets_ewk_t2


python ROC_curve.py -s ~/ewk_val_run7.npy -b ~/w+jets_val_run7.npy  \
                        -m ../models/run7/w+jets_v5.yaml  -o ../ROC_curves/run7/w+jets  -f roc_model_w+jets_ewk_w+jets

