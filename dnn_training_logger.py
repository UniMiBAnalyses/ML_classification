'''
keras callback to plot loss
'''

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import keras
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

from scipy import stats

def ks_w2(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    return np.max(np.abs(cdf1we - cdf2we))
## callbacks
# updatable plot
# a minimal example (sort of)

class TrainingLogger(keras.callbacks.Callback):
    def __init__(self, model, data, dnncut=0.85, batch_mode=False):
        self.model = model
        self.X_train = data["X_train"]
        self.X_test = data["X_val"]
        self.y_train = data["y_train"]
        self.y_test = data["y_val"]
        self.W_train = data["W_train"]
        self.W_test = data["W_val"]  # use the validation data for plots
        # Weight Not Normalized
        self.Wnn_train = data["Wnn_train"]
        self.Wnn_test = data["Wnn_val"]
        self.batch_mode = batch_mode
        self.dnncut = dnncut
        self.signal_mask_train = self.y_train == 1
        self.signal_mask_test = self.y_test == 1
        self.bkg_mask_train = self.y_train == 0
        self.bkg_mask_test = self.y_test == 0
        self.tot_bkg_train = self.Wnn_train[self.bkg_mask_train].sum()
        self.tot_bkg_test = self.Wnn_test[self.bkg_mask_test].sum()
        self.tot_sig_train = self.Wnn_train[self.signal_mask_train].sum()
        self.tot_sig_test = self.Wnn_test[self.signal_mask_test].sum()

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.dnn_score_plot = []
        self.dnn_score_log = []
        self.kstest_sig = []
        self.kstest_bkg = []
        self.significance_total = []
        self.precision_test = []
        self.precision_train = []
        self.recall_test = []
        self.recall_train = []
        self.f1_test = []
        self.f1_train = []
        self.pred_train=[]
        self.pred_test=[]
        self.auc_points_train = [ ]
        self.auc_points_test = [ ]
        self.figure = None
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.performance_save(logs)
        if not self.batch_mode:
            self.performance_plot()

    def performance_save(self, logs):
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss')) #training_loss[0])
        self.val_loss.append(logs.get('val_loss'))
        # 'acc' and 'val_acc' work in "96 python3" in swan.cern.ch
        self.acc.append(logs.get('accuracy')) #training_loss[1])
        self.val_acc.append(logs.get('val_accuracy'))
        # in newer keras these may be 'accuracy' and 'val_accuracy'
        self.i += 1

        self.pred_test = self.model.predict(self.X_test, batch_size=4096)
        self.pred_train = self.model.predict(self.X_train, batch_size=4096)
        self.pred_test = np.array(self.pred_test).flatten()
        self.pred_train = np.array(self.pred_train).flatten()
        
        # auc_w_test = roc_auc_score(self.y_test, self.pred_test, sample_weight=self.W_test)
        # self.auc_test.append(auc_w_test)
        
        # auc_w_train = roc_auc_score(self.y_train, self.pred_train, sample_weight=self.W_train)
        # self.auc_train.append(auc_w_train)

        kstest_pval_sig = stats.ks_2samp(self.pred_train[self.signal_mask_train], self.pred_test[self.signal_mask_test], mode="asymp") # (statistics, pvalue)
        self.kstest_sig.append(kstest_pval_sig[1])
        kstest_pval_bkg = stats.ks_2samp(self.pred_train[self.bkg_mask_train], self.pred_test[self.bkg_mask_test], mode="asymp") # (statistics, pvalue)
        self.kstest_bkg.append(kstest_pval_bkg[1])
        # print("KS test (dnn output: sig (train) vs sig (val))", kstest_pval_sig, ". good: ", kstest_pval_sig[1] > 0.05)
        # print("KS test (dnn output: bkg (train) vs bkg (val))", kstest_pval_bkg, ". good: ", kstest_pval_bkg[1] > 0.05)

        TP_train = self.Wnn_train[(self.signal_mask_train) & (self.pred_train > self.dnncut)].sum()
        FP_train = self.Wnn_train[(self.bkg_mask_train) & (self.pred_train > self.dnncut)].sum() 
        #T_train = self.Wnn_train[(self.signal_mask_train)].sum()
    
        TP_test = self.Wnn_test[(self.signal_mask_test) & (self.pred_test > self.dnncut)].sum()
        FP_test = self.Wnn_test[(self.bkg_mask_test) & (self.pred_test > self.dnncut)].sum()
        #T_test = self.Wnn_test[(self.signal_mask_test)].sum()

        #self.significance_total.append( (TP_train+TP_test)/np.sqrt( FP_train+FP_test ) )

        self.precision_test.append(TP_test/ (TP_test+FP_test))
        self.precision_train.append(TP_train/ (TP_train+FP_train))
        self.recall_test.append(TP_test/ self.tot_sig_test)
        self.recall_train.append(TP_train/ self.tot_sig_train)

        self.f1_train.append( 2* (self.precision_train[-1] * self.recall_train[-1])/ (self.precision_train[-1] + self.recall_train[-1]) )
        self.f1_test.append( 2* (self.precision_test[-1] * self.recall_test[-1])/ (self.precision_test[-1] + self.recall_test[-1]) )


        # 10 points ROC curve
        thr = np.linspace(self.dnncut, 1.,15)
        b_train = [] 
        b_test = [] 
        s_train = []
        s_test = []

        for x in thr:
            tp_train =   self.Wnn_train[(self.signal_mask_train) & (self.pred_train > x)].sum()
            tp_test =   self.Wnn_test[(self.signal_mask_test) & (self.pred_test > x)].sum()
            fp_train =   self.Wnn_train[(self.bkg_mask_train) & (self.pred_train > x)].sum()
            fp_test =   self.Wnn_test[(self.bkg_mask_test) & (self.pred_test > x)].sum()
           
            b_train.append(fp_train/self.tot_bkg_train)
            b_test.append(fp_test/self.tot_bkg_test)
            s_train.append(tp_train/self.tot_sig_train)
            s_test.append(tp_test/self.tot_sig_test)
        
        # Compute significance for the last 3 points
        self.significance_total.append([
            ( (s_train[0]*self.tot_sig_train +s_test[0]*self.tot_sig_test)/np.sqrt( b_train[0]*self.tot_bkg_train+b_test[0]*self.tot_bkg_test ),  thr[0]),
            ( (s_train[4]*self.tot_sig_train +s_test[4]*self.tot_sig_test)/np.sqrt( b_train[4]*self.tot_bkg_train+b_test[4]*self.tot_bkg_test ),  thr[4]),
            ( (s_train[7]*self.tot_sig_train +s_test[7]*self.tot_sig_test)/np.sqrt( b_train[7]*self.tot_bkg_train+b_test[7]*self.tot_bkg_test ),  thr[7])
        ])
        
        self.auc_points_test.append((s_test, b_test))
        self.auc_points_train.append((s_train, b_train))

    def performance_plot(self):
        self.figure, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24,16), dpi=150)

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.loss, "o-", label="loss (train)")
        ax1.plot(self.x, self.val_loss, "o-", label="loss (val)")
        ax1.set_yscale("log")
        ax1.set_xlabel("epochs")
        ax1.legend()
        ax1.set_title("Loss")

       
        bins=30
        ax2.hist(self.pred_train[self.bkg_mask_train],weights=self.W_train[self.bkg_mask_train], bins=bins, range=(0.,1.), density=True, label="bkg (train)", histtype="step")
        ax2.hist(self.pred_train[self.signal_mask_train],weights=self.W_train[self.signal_mask_train], bins=bins, range=(0.,1.), density=True, label="sig (train)", histtype="step")
        ax2.hist(self.pred_test[self.bkg_mask_test],weights=self.W_test[self.bkg_mask_test], bins=bins, range=(0.,1.), density=True, label="bkg (val)", histtype="step")
        ax2.hist(self.pred_test[self.signal_mask_test],weights=self.W_test[self.signal_mask_test], bins=bins, range=(0.,1.), density=True, label="sig (val)", histtype="step")
        ax2.set_xlabel("DNN output")
        ax2.legend(loc='upper center')
        ax2.set_title("DNN shape")

        s1 = [ s[0][0] for s in self.significance_total ]
        s2 = [ s[1][0] for s in self.significance_total ]
        s3 = [ s[2][0] for s in self.significance_total ]
        ax3.plot(self.x, s1, "o-",label="S/sqrt(B) thr={:.2f}".format(self.significance_total[0][0][1]))
        ax3.plot(self.x, s2, "o-", label="S/sqrt(B) thr={:.2f}".format(self.significance_total[0][1][1]))
        ax3.plot(self.x, s3, "o-", label="S/sqrt(B) thr={:.2f}".format(self.significance_total[0][2][1]))
        ax3.set_ylabel("S / sqrt(B)")
        ax3.set_xlabel("epochs")
        ax3.legend()
        ax3.set_title("Significance")


        # ax2.plot(self.x, self.acc, "o-", label="accuracy (train)")
        # ax2.plot(self.x, self.val_acc, "o-", label="accuracy (val)")
        # ax2.set_xlabel("epochs")
        # ax2.legend()
        ax4.plot(self.x, self.f1_train, "o-", label="F1 score (train) thr={}".format(self.dnncut))
        ax4.plot(self.x, self.f1_test, "o-", label="F1 score (test) thr={}".format(self.dnncut))
        ax4.set_xlabel("epochs")
        ax4.legend()
        ax4.set_title("F1 score")
        
        ax5.plot(self.x, self.precision_train, "o-", label="Precision (train) thr={}".format(self.dnncut))
        ax5.plot(self.x, self.precision_test, "o-", label="Precision (test) thr={}".format(self.dnncut))
        ax5.set_xlabel("epochs")
        ax5.legend()
        ax5.set_title("Precision")


        ax6.plot(self.x, self.recall_train, "o-", label="Recall (train) thr={}".format(self.dnncut))
        ax6.plot(self.x, self.recall_test, "o-", label="Recall (test) thr={}".format(self.dnncut))
        ax6.set_xlabel("epochs")
        ax6.legend()
        ax6.set_title("Recall")

        ax7.plot(self.x, self.kstest_sig, "o-", label="sig (train) vs sig (val). kstest pval")
        ax7.plot(self.x, self.kstest_bkg, "o-", label="bkg (train) vs bkg (val). kstest pval")
        ax7.plot((self.x[0], self.x[-1]), (0.05, 0.05), 'k-')
        ax7.legend()
        ax7.set_xlabel("epochs")
        ax7.set_yscale('log')
        ax7.set_title("KS test")


        if len(self.auc_points_test)>1:
            ax8.plot(self.auc_points_test[-2][1],self.auc_points_test[-2][0],  linestyle="--",  label="ROC test (prev)" , color="green")        
        ax8.plot(self.auc_points_train[-1][1],self.auc_points_train[-1][0], label="ROC train",color="blue" )
        ax8.plot(self.auc_points_test[-1][1],self.auc_points_test[-1][0], label="ROC test",color="orange" )
        ax8.set_ylabel("Signal efficiency")
        ax8.set_xlabel("Background contamination")
        ax8.legend()
        ax8.set_title("ROC curve")

        for ia in range(1, 7):
            if len(self.auc_points_train) >= ia:
                auc  = self.auc_points_train[-ia]
                if ia >1:
                    ax9.plot(auc[1],auc[0],  linestyle="--",  label="ROC train iter=-{}".format(ia-1))
                if ia == 1:
                    ax9.plot(auc[1],auc[0], linewidth=2,  label="ROC train latest".format(ia), color='blue')      
        ax9.set_ylabel("Signal efficiency")
        ax9.set_xlabel("Background contamination")
        ax9.legend()
        ax9.set_title("ROC curve history")
        
        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()

    def save_figure(self, fname):
        if self.batch_mode:
            self.performance_plot()
        self.figure.savefig(fname)
        plt.close(self.figure)

    def save_logs(self,file):
        df = pd.DataFrame()
        df["loss"] = self.loss
        df["val_loss"] = self.val_loss
        df["acc"] = self.acc
        df["kstest_sig"] = self.kstest_sig
        df["kstest_bkg"] = self.kstest_bkg
        df["significance_total"] = self.significance_total
        df["precision_test"] = self.precision_test
        df["precision_train"] = self.precision_train
        df["recall_test"] = self.recall_test
        df["recall_train"] = self.recall_train
        df["f1_test"] = self.f1_test
        df["f1_train"] = self.f1_train
        df["auc_points_train"] = self.auc_points_train
        df["auc_points_test"] = self.auc_points_test
        df.to_csv(file, index=False, sep=',')