'''
keras callback to plot loss
'''

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import keras
from sklearn.metrics import roc_auc_score, roc_curve

from scipy import stats

## callbacks
# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model, data):
        self.model = model
        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]
        self.W_train = data["W_train"]
        self.W_test = data["W_test"]


    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.auc = []
        self.dnn_score_plot = []
        self.dnn_score_log = []
        self.kstest_log = []
        self.figure = None
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        #training_loss = model.predict_generator(training_generator, steps=1000)
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss')) #training_loss[0])
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc')) #training_loss[1])
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        self.figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24,8))
        clear_output(wait=True)

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, "o-", label="loss")
        ax1.plot(self.x, self.val_losses, "o-", label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, "o-", label="accuracy")
        ax2.plot(self.x, self.val_acc, "o-", label="validation accuracy")
        ax2.legend()
        
        pred_test = self.model.predict(self.X_test, batch_size=2048)
        auc_w = roc_auc_score(self.y_test, pred_test, sample_weight=self.W_test)
        self.auc.append(auc_w)
        ax3.plot(self.x, self.auc, "o-", label="auc - test")
        ax3.legend()

        pred_train = self.model.predict(self.X_train, batch_size=2048)
        bins=25
        ax4.hist(pred_train[self.y_train==0],weights=self.W_train[self.y_train==0], bins=bins, density=True, label="false - train", histtype="step")
        ax4.hist(pred_train[self.y_train==1],weights=self.W_train[self.y_train==1], bins=bins, density=True, label="true - train", histtype="step")
        dnnout_false = ax4.hist(pred_test[self.y_test==0],weights=self.W_test[self.y_test==0], bins=bins,density=True, label="false - test", histtype="step")
        dnnout_true  = ax4.hist(pred_test[self.y_test==1],weights=self.W_test[self.y_test==1], bins=bins, density=True, label="true - test", histtype="step")
        ax4.legend()

        rtest  = [x[0] for x in pred_test[self.y_test==1]]
        rtrain = [x[0] for x in pred_train[self.y_train==1]]
        kstest_pval = stats.ks_2samp(rtrain, rtest) # (statistics, pvalue)
        self.kstest_log.append(kstest_pval[1])
        print("KS test (dnn output: true train vs true temp)", kstest_pval, ". good: ", kstest_pval[1] > 0.05)
        
        ax5.hist(dnnout_false[1][:-1], bins=dnnout_false[1], weights= (dnnout_true[0] / dnnout_false[0]) , label="dnn out ratio")
        ax5.legend()
        self.dnn_score_log.append( (dnnout_true[0][dnnout_true[1][:-1] > 0.8] / dnnout_false[0][dnnout_true[1][:-1] > 0.8]).sum() )
        ax6.plot(self.x, self.dnn_score_log, "o-", label="sum bincontent dnnout ratio > 0.8")
        #ax6.legend()
        ax6.set_ylabel("sum bincontent dnnout ratio > 0.8", color='blue')
        ax7 = ax6.twinx()
        ax7.plot(self.x, self.kstest_log, "o-", label="kstest pval", color='orange')
        #ax7.legend()
        ax7.set_yscale('log')
        ax7.set_ylabel('kstest pval', color='orange')
        
        self.dnn_score_plot.append((dnnout_false,dnnout_true))
        
        plt.show()

    def save_figure(self, fname):
        self.figure.savefig(fname)