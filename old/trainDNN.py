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
parser.add_argument('-sv','--signal-validation', type=str, required=True,
                     help="Dataset for validation (signal)")
parser.add_argument('-b', '--background', type=str, required=True)
parser.add_argument('-bv','--background-validation', type=str, required=True,
                     help="Dataset for validation (bkg)")
parser.add_argument('-e', '--epochs', type=int, required=False, default=10)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-ms', '--model-schema', type=str, required=True, help="Model structure")
parser.add_argument('-vs', '--validation-samples', type=int, required=False, default=1e5, 
                    help="Number of validation samples")
parser.add_argument('-i', '--initial-epoch', type=int, required=False, default=0)
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=50)
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-3)
parser.add_argument('-dr', '--decay-rate', type=float, required=False, default=0)
parser.add_argument('-p', '--patience', type=str, required=False, default="0.0001:5",
                    help="Patience format:  delta_val:epochs")
parser.add_argument('-m', '--model', type=str, required=False)
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-sst', '--save-steps', action="store_true")
args = parser.parse_args()

if not os.path.exists(args.output+"/"):
    os.makedirs(args.output)

print(">>> Loading datasets...")
sig = np.load(args.signal)
sig_val = np.load(args.signal_validation)
bkg = np.load(args.background)
bkg_val = np.load(args.background_validation)

# Balance datasets to have 50/50 sig/bkg
n_events = min(sig.shape[0], bkg.shape[0])
n_val = args.validation_samples

labels_train = np.append( np.ones(n_events), np.zeros(n_events))
data_train = np.append(sig[:n_events], bkg[:n_events], axis = 0)

labels_val = np.append(np.ones(n_val), np.zeros(n_val))
data_val = np.append(sig_val[:n_val], bkg_val[:n_val], axis = 0)

# Extract weights
W_train = data_train[:,0]
data_train = data_train[:,1:]
W_val = data_val[:,0]
data_val = data_val[:,1:]

print(">>> Normalising dataset...")
scaler = StandardScaler()
scaler.fit(data_train)

data_train = scaler.transform(data_train)
data_val = scaler.transform(data_val)

joblib.dump(scaler, args.output+ "/scaler.pkl")


if args.model == None: 
    model = getModel(args.model_schema, data_train.shape[1])
else:
    print(">>> Loading model ({0})...".format(args.model))
    model = load_model(args.model)

if not args.evaluate:
    # Training procedure
    if args.save_steps:
        auto_save = ModelCheckpoint(args.output +"/current_model_epoch{epoch:02d}", monitor='val_loss', 
                    verbose=1, save_best_only=False, save_weights_only=False, 
                    mode='auto', period=1)
    else:
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

    history = model.fit(data_train, labels_train,
                        validation_data = (data_val, labels_val, W_val),                        
                        epochs=args.epochs, initial_epoch=args.initial_epoch,
                        batch_size=args.batch_size, shuffle=True, 
                        sample_weight = W_train,
                        callbacks=[auto_save, early_stop, lr_sched, csv_logger])

# Evalutation
print(">>> Computing AUC...")

from sklearn.metrics import roc_auc_score, roc_curve

pred = model.predict(data_val,batch_size=2048)
auc = roc_auc_score(labels_val, pred, sample_weight=W_val)
print("AUC score: " + str(auc))

print(">>> Saving ROC curve...")
fp , tp, th = roc_curve(labels_val, pred, sample_weight=W_val)
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
    f.write("model_schema: {0}\n".format(args.model_schema))
    f.write("batch_size: {0}\n".format(args.batch_size))
    f.write("learning_rate: {0}\n".format(args.learning_rate))
    f.write("decay_rate: {0}\n".format(args.decay_rate))
    f.write("patience: {0}\n".format(args.patience))
    f.write("AUC: {0}\n".format(auc))
    f.close()