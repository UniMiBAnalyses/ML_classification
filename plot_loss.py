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
        self.auc_train = []
        self.auc_test = []
        self.dnn_score_plot = []
        self.dnn_score_log = []
        self.kstest_log = []
        self.significance_test = []
        self.significance_train = []
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
        self.figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24,10))
        clear_output(wait=True)

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, "o-", label="loss (train)")
        ax1.plot(self.x, self.val_losses, "o-", label="loss (val)")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, "o-", label="accuracy (train)")
        ax2.plot(self.x, self.val_acc, "o-", label="accuracy (val)")
        ax2.legend()
        
        pred_test = self.model.predict(self.X_test, batch_size=2048)
        auc_w_test = roc_auc_score(self.y_test, pred_test, sample_weight=self.W_test)
        self.auc_test.append(auc_w_test)
        pred_train = self.model.predict(self.X_train, batch_size=2048)
        auc_w_train = roc_auc_score(self.y_train, pred_train, sample_weight=self.W_train)
        self.auc_train.append(auc_w_train)
        ax3.plot(self.x, self.auc_train, "o-", label="auc (train)")
        ax3.plot(self.x, self.auc_test, "o-", label="auc (val)")
        ax3.legend()

        # pred_train = self.model.predict(self.X_train, batch_size=2048)
        bins=25
        ax4.hist(pred_train[self.y_train==0],weights=self.W_train[self.y_train==0], bins=bins, range=(0.,1.), density=True, label="bkg (train)", histtype="step")
        ax4.hist(pred_train[self.y_train==1],weights=self.W_train[self.y_train==1], bins=bins, range=(0.,1.), density=True, label="sig (train)", histtype="step")
        dnnout_false = ax4.hist(pred_test[self.y_test==0],weights=self.W_test[self.y_test==0], bins=bins,density=True, label="bkg (val)", histtype="step")
        dnnout_true  = ax4.hist(pred_test[self.y_test==1],weights=self.W_test[self.y_test==1], bins=bins, density=True, label="sig (val)", histtype="step")
        ax4.legend()

        rtest  = [x[0] for x in pred_test[self.y_test==1]]
        rtrain = [x[0] for x in pred_train[self.y_train==1]]
        kstest_pval = stats.ks_2samp(rtrain, rtest) # (statistics, pvalue)
        self.kstest_log.append(kstest_pval[1])
        print("KS test (dnn output: sig (train) vs sig (val))", kstest_pval, ". good: ", kstest_pval[1] > 0.05)
        ax5.plot(self.x, self.kstest_log, "o-", label="sig (train) vs sig (val). kstest pval")
        ax5.plot((self.x[0], self.x[-1]), (0.05, 0.05), 'k-')
        ax5.legend()
        ax5.set_yscale('log')

        #print(self.y_train.shape, self.y_train[self.y_train==1].shape, self.y_train[self.y_train==0].shape,)
        #print(self.X_train.shape, )
        # s_great_train_mask = (self.y_train==1) & (pred_train[self.y_train==1] > 0.8)
        pred_train = pred_train.flatten()
        pred_test = pred_test.flatten()
        print("train", self.X_train.shape, self.y_train.shape, self.W_train.shape, pred_train.shape )
        #print("pred", pred_train[self.y_train==1].shape, len(pred_train[self.y_train==1]) )
        #print("W", self.W_train[self.y_train==1].shape)
        #s_tot = np.zeros(len(pred_train))
        dnnout_cut = 0.8
        s_geq_train = np.ones(len(self.y_train[(self.y_train==1) & (pred_train > dnnout_cut)])) * self.W_train[(self.y_train==1) & (pred_train > dnnout_cut)]
        b_geq_train = np.ones(len(self.y_train[(self.y_train==0) & (pred_train > dnnout_cut)])) * self.W_train[(self.y_train==0) & (pred_train > dnnout_cut)]
        significance_train = ( s_geq_train.sum() ) / (np.sqrt( b_geq_train.sum() ))
        self.significance_train.append(significance_train)
        ax6.plot(self.x, self.significance_train, "o-", color="blue")
        ax6.set_ylabel("S / sqrt(B) (train)", color='blue')
        s_geq_test = np.ones(len(self.y_test[(self.y_test==1) & (pred_test > dnnout_cut)])) * self.W_test[(self.y_test==1) & (pred_test > dnnout_cut)]
        b_geq_test = np.ones(len(self.y_test[(self.y_test==0) & (pred_test > dnnout_cut)])) * self.W_test[(self.y_test==0) & (pred_test > dnnout_cut)]
        significance_test = ( s_geq_test.sum() ) / (np.sqrt( b_geq_test.sum() ))
        self.significance_test.append(significance_test)
        ax7 = ax6.twinx()
        ax7.plot(self.x, self.significance_test, "o-", color="orange")
        ax7.set_ylabel("S / sqrt(B) (val)", color='orange')

        plt.show()

    def save_figure(self, fname):
        self.figure.savefig(fname)