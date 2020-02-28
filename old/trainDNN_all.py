import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from models import *
from keras.callbacks import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signal', type=str, required=True)
parser.add_argument('-sv', '--signal-validation', type=str, required=True)
parser.add_argument('-b', '--backgrounds', nargs='+', type=str, required=True)
parser.add_argument('-bv', '--backgrounds-validation', nargs='+', type=str, required=True)
parser.add_argument('-e', '--epochs', type=int, required=False, default=10)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-i', '--initial-epoch', type=int, required=False, default=0)
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=50)
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-3)
parser.add_argument('-dr', '--decay-rate', type=float, required=False, default=0)
parser.add_argument('-ms', '--model-schema', type=str, required=True, help="Model structure")
parser.add_argument('-p', '--patience', type=str, required=False, default="0.001:5",
                    help="Patience format:  delta_val:epochs")
parser.add_argument('-m', '--model', type=str, required=False)
parser.add_argument('-vs', '--validation-samples', type=int, required=False, default=1e5, 
                    help="Number of validation samples")
parser.add_argument('-ev', '--evaluate', action="store_true")
args = parser.parse_args()

if not os.path.exists(args.output+"/"):
    os.makedirs(args.output)

nval = args.validation_samples

print(">>> Loading dataset...")
sig = np.load(args.signal)
sig_val = np.load(args.signal_validation)[:nval*4]
bkgs = []
bkgs_val = []
for bk,bkval in zip(args.backgrounds, args.backgrounds_validation):
    bkgs.append(np.load(bk))
    bkgs_val.append(np.load(bkval)[:nval])

datasets = [sig] + bkgs
datasets_val = [sig_val] + bkgs_val

#training
labels = None
for i, d in enumerate(datasets):
    n_events = d.shape[0]
    print("Dataset {} : {}".format(i, n_events))
    ones = np.ones(int(n_events))
    zeros = np.zeros(int(n_events))
    ls = []
    for j in range(len(datasets)):
        if j == i: 
            ls.append(ones)
        else:
            ls.append(zeros)
    if i == 0:
        labels = np.stack(ls, axis=1)
    else:
        labels = np.append(labels, np.stack(ls, axis=1), axis=0 )

data_train = np.concatenate(datasets, axis = 0)

# validation
labels_val = None
for i, d in enumerate(datasets_val):
    ones = np.ones(int(d.shape[0]))
    zeros = np.zeros(int(d.shape[0]))
    ls = []
    for j in range(len(datasets_val)):
        if j == i: 
            ls.append(ones)
        else:
            ls.append(zeros)
    if i == 0:
        labels_val = np.stack(ls, axis=1)
    else:
        labels_val = np.append(labels_val, np.stack(ls, axis=1), axis=0 )

data_val = np.concatenate(datasets_val, axis = 0)

# Extract weights
W_train = data_train[:,0]
X_train = data_train[:,1:]
W_val = data_val[:,0]
X_val = data_val[:,1:]

print(">>> Normalising dataset...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

joblib.dump(scaler, args.output+ "/scaler.pkl")


if args.model == None: 
    model = getModel(args.model_schema, X_train.shape[1])
else:
    print(">>> Loading model ({0})...".format(args.model))
    model = load_model(args.model)


if not args.evaluate:
    # Training procedure
    auto_save = ModelCheckpoint(args.output +"/current_model", monitor='val_loss', 
                    verbose=1, save_best_only=True, save_weights_only=False, 
                    mode='auto', period=2)

    min_delta = float(args.patience.split(":")[0])
    p_epochs = int(args.patience.split(":")[1])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, 
                                patience=p_epochs, verbose=1)

    def reduceLR (epoch):
        return args.learning_rate * (1 / (1 + epoch*args.decay_rate))

    lr_sched = LearningRateScheduler(reduceLR, verbose=1)

    csv_logger = CSVLogger(args.output +'/training.log')


    print(">>> Training...")

    history = model.fit(X_train, labels, 
                        validation_data=(X_val, labels_val, W_val),
                        epochs=args.epochs, initial_epoch=args.initial_epoch,
                        batch_size=args.batch_size, 
                        sample_weight = W_train,
                        validation_split=0.2, shuffle=True,
                        callbacks=[auto_save, lr_sched, csv_logger, early_stop])

#Evalutation
print(">>> Computing AUC of signal...")

from sklearn.metrics import roc_auc_score, roc_curve, auc
pred = model.predict(X_val)
auc = roc_auc_score(labels_val[:,0], pred[:,0])
print("AUC score: " + str(auc))

print(">>> Saving ROC curve...")
fp , tp, th = roc_curve(labels_val[:,0], pred[:,0])
r = open(args.output + "/roc_curve.txt", "w")
r.write("br,se,th\n")
bkg_rej = 1 - fp
for i in range(len(fp)):
    r.write("{0},{1},{2}\n".format(bkg_rej[i], tp[i], th[i]))
r.close()


if not args.evaluate:
    print(">>> Saving parameters...")
    f = open(args.output + "/configs.txt", "w")
    f.write("epochs: {0}\n".format(args.epochs))
    f.write("batch_size: {0}\n".format(args.batch_size))
    f.write("learning_rate: {0}\n".format(args.learning_rate))
    f.write("decay_rate: {0}\n".format(args.decay_rate))
    f.write("model_schema: {0}\n".format(args.model_schema))
    f.write("AUC: {0}\n\n".format(auc))

    # Print the number of training samples
    for i, d in enumerate(datasets):
        n_events = d.shape[0]
        f.write("Dataset {} : {}\n".format(i, n_events))
    
    f.close()