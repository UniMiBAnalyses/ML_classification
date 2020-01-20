'''
This script plot ROC curves for every sample. 
It needs the basedir containing all the roc curves data. 
'''
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rc('figure', figsize=(10,8), dpi=100)


colors= {
    "qcdww": (1, 0.4, 0),
    "singleTop": (0.4, 0.8, 1),
    "ttbar": (1, 0.8, 0.2),
    "wjets": (0.8, 0.2, 0.2),
    "ewk": (0., 0.4, 1)
}

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir', type=str, required=True)
parser.add_argument('-o', '--output-dir', type=str, required=True)
args = parser.parse_args()

base = args.input_dir
samples = [("t0", "QCD-WW"), ("t1", "singleTop"), ("t2", "ttbar"),( "w+jets", "W+Jets")]

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for sample, title in samples:
    roc0 = pd.read_csv(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t0", skiprows=1 )
    roc1 = pd.read_csv(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t1", skiprows=1 ) 
    roc2 = pd.read_csv(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t2", skiprows=1  )
    roc3 = pd.read_csv(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_w+jets", skiprows=1  ) 
    with open(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t0") as file:
        auc0 = float(file.readline()[5:])
    with open(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t1") as file:
        auc1 = float(file.readline()[5:])
    with open(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_t2") as file:
        auc2 = float(file.readline()[5:])
    with open(base +"/" + sample+ "/roc_model_"+ sample +"_ewk_w+jets") as file:
        auc3 = float(file.readline()[5:])
    plt.figure(figsize=(5,4), dpi=300)
    plt.plot(1- roc0["br"], roc0["se"],label="AUC={:.3f} - NN QCD-WW".format(auc0), c=colors["qcdww"])
    plt.plot(1- roc1["br"], roc1["se"],label="AUC={:.3f} - NN singleTop".format(auc1),c=colors["singleTop"])
    plt.plot(1- roc2["br"], roc2["se"],label="AUC={:.3f} - NN ttbar".format(auc2),c=colors["ttbar"])
    plt.plot(1- roc3["br"], roc3["se"],label="AUC={:.3f} - NN W+jets".format(auc3),c=colors["wjets"])
    plt.legend()
    plt.title("ROC - {} classifier".format(title))
    plt.ylabel("signal efficiency")
    plt.xlabel("mis-id probability")
    plt.savefig(args.output_dir + "/" + title+ ".pdf")
    plt.clf()