import numpy as np
import os
import yaml
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signal', type=str, required=True)
parser.add_argument('-b', '--background', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-f', '--filename', type=str, required=True, help="filename")
parser.add_argument("-ss", "--score-size", type=int, required=False,
                    default=1, help="Size of the score vector")
args = parser.parse_args()

model_conf = yaml.load(open(args.model))

if not os.path.exists(args.output):
    os.makedirs(args.output)

print(">>> Loading dataset...")
# Remove first columns with event weight
sig = np.load(args.signal)
bkg = np.load(args.background)

# Balance datasets to have 50/50 sig/bkg
n_events = min(sig.shape[0], bkg.shape[0])

labels= np.append( np.ones(n_events), np.zeros(n_events))
data = np.append(sig[:n_events], bkg[:n_events], axis = 0)

# Extract sample weights
weights = data[:,0]
data = data[:,1:]

print(">>> Normalising dataset...")
scaler = joblib.load("../../"+model_conf["scaler"])
data_rescaled = scaler.fit_transform(data)

from keras.models import load_model

print(">>> Loading model ({0})...".format(args.model))
model = load_model("../../"+model_conf["model_path"])

# Evalutation
print(">>> Predict...")
pred = model.predict(data_rescaled,batch_size=2048)


from sklearn.metrics import roc_auc_score, roc_curve
print(">>> Computing AUC...")
if args.score_size == 1:
    auc = roc_auc_score(labels, pred, sample_weight=weights)
else:
    # Take the first class for the signal
    auc = roc_auc_score(labels, pred[:,0],sample_weight=weights)
print("AUC score: " + str(auc))

print(">>> Saving ROC curve...")
if args.score_size == 1:
    fp , tp, th = roc_curve(labels, pred, sample_weight=weights)
else:
    fp , tp, th = roc_curve(labels, pred[:,0], sample_weight=weights)

r = open(args.output + "/"+ args.filename, "w")
r.write("#AUC={0}\n".format(auc))
r.write("br,se,th\n")
bkg_rej = 1 - fp
for i in range(len(fp)):
    r.write("{0},{1},{2}\n".format(bkg_rej[i], tp[i], th[i]))
r.close()

