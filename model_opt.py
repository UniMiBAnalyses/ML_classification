import plot_loss

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, BatchNormalization, Dropout
import pickle

import numpy as np
import yaml
import time
from copy import deepcopy
import os
import datetime

import logging
logging.basicConfig(level=logging.INFO)


class VbsDnn():
    def __init__ (self,
        config, 
        input_dim,
        batch_size,
        n_layers,
        n_nodes,
        dropout,
        batch_norm,
        epochs=300,
        test_ratio=0.1,
        val_ratio=0.2,
        patience = (0.001, 10),
        ):

        self._config = deepcopy(config)
        self.__input_dim = input_dim
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__model_dir = None
        self.__n_layers = n_layers
        self.__n_nodes = n_nodes
        self.__dropout = dropout
        self.__batch_norm = batch_norm
        self.__test_ratio = test_ratio
        self.__val_ratio = val_ratio
        self.__patience = patience

        self._data_split, self._generators = self.data_loader()

        self._model = self.get_model()

        self._train_monitor = plot_loss.PlotLosses(self._model, self._data_split, batch_mode=True)

        self._history = None

    def data_loader(self):
        '''
        This can be used also to evaluate the input variables with AVOVA F-test
        '''
        ## read data from files
        logging.debug("Loading data")
        config_base_dir = os.path.join(self._config["base_dir"], self._config["plot_config"])

        # create the model directory
        utc_seconds = str(datetime.datetime.now().timestamp()).split(".")[0]
        logging.info(utc_seconds)
        self.__model_dir   = os.path.join(config_base_dir, self._config["cut"] , "optimization", utc_seconds)
        os.makedirs(self.__model_dir, exist_ok=True)

        # load numpy
        samples_dir = os.path.join(config_base_dir, self._config["cut"] , "samples", self._config["samples_version"])
        signal = pickle.load(open(os.path.join(samples_dir, "for_training/signal_balanced.pkl"),     "rb"))
        bkg    = pickle.load(open(os.path.join(samples_dir, "for_training/background_balanced.pkl"), "rb"))

        # Keep only the first "input-dim" columns
        self._config["cols"] = self._config["cols"][:self.__input_dim]
        logging.debug(self._config["cols"])

        ## create numpy arrays
        X_sig = signal[self._config["cols"]].values
        X_bkg = bkg[self._config["cols"]].values
        Y_sig = np.ones(len(X_sig))
        Y_bkg = np.zeros(len(X_bkg))
        W_sig = (signal["weight_norm"]).values
        W_bkg = (bkg["weight_norm"]).values
        Wnn_sig = (signal["weight_"]).values
        Wnn_bkg = (bkg["weight_"]).values

        X = np.vstack([X_sig, X_bkg])
        Y = np.hstack([Y_sig, Y_bkg])
        W = np.hstack([W_sig, W_bkg])
        Wnn = np.hstack([Wnn_sig, Wnn_bkg])

        ## import scaler configuration
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pickle.dump(scaler, open(f"{self.__model_dir}/scaler_model.pkl", "wb"))

        ## Balance
        X_train, X_test, y_train, y_test, W_train, W_test , Wnn_train, Wnn_test = train_test_split(X_scaled, Y,  W, Wnn,    
                                        test_size=self.__test_ratio, random_state=42, stratify=Y)
        X_train, X_val,  y_train,  y_val, W_train, W_val,   Wnn_train, Wnn_val =  train_test_split(X_train, y_train, W_train, Wnn_train, 
                                        test_size=self.__val_ratio, random_state=42,  stratify=y_train) 

        data_split = {
            "X_train": X_train,
            "X_test" : X_test, 
            "X_val":   X_val,
            "y_train": y_train,
            "y_test" : y_test,
            "y_val"  : y_val,
            "W_train": W_train,
            "W_test" : W_test,
            "W_val" : W_val,
            "Wnn_train": Wnn_train,
            "Wnn_test" : Wnn_test, 
            "Wnn_val" : Wnn_val, 
        }

        ## Oversampling
        training_generator,   steps_per_epoch_train = balanced_batch_generator(X_train, y_train, W_train, batch_size=self.__batch_size, sampler=RandomOverSampler())
        #validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_val,   y_val,   W_val,   batch_size=self.__batch_size, sampler=RandomOverSampler()) ## test != val
        validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_val,  y_val,  W_val,   batch_size=self.__batch_size, sampler=RandomOverSampler()) ## test == val

        generators = {
            "training_generator": training_generator,
            "steps_per_epoch_train": steps_per_epoch_train,
            "validation_generator": validation_generator,
            "steps_per_epoch_val": steps_per_epoch_val,
        }

        return data_split, generators

    def get_model(self):
        logging.debug("Creating model")
        model = Sequential()
        for ilay in range(self.__n_layers):
            if ilay==0:
                model.add(Dense(self.__n_nodes, input_dim=self.__input_dim))
            else:
                model.add(Dense(self.__n_nodes))
            if self.__batch_norm:
                model.add(BatchNormalization())
            model.add(Activation("relu"))
            if self.__dropout > 0:
                model.add(Dropout(self.__dropout))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        return model

    def save_metadata(self):
        ## SAVE THE MODEL, ITS METADATA AND TRAINING INFORMATIONS

        # Retrieve the model metadata that should be saved
        # dump the variables list
        for v in self.__dict__:
            if v.startswith("_"+self.__class__.__name__+"__"):
                self._config[v.split("__")[1]] = getattr(self, v)
            
        # dump the config
        model_config_file = os.path.join(self.__model_dir, "model_config.yml")
        if os.path.isfile(model_config_file):
            print("ACHTUNG! model_config_file file already existing: old file renamed with '_old'")
            os.rename(model_config_file, model_config_file[:-4] + "_old.yml")
        with open(model_config_file, "w") as out_var_file:
            out_var_file.write(yaml.dump(self._config))  

        # save figure with training summary
        self._train_monitor.save_figure( os.path.join(self.__model_dir, "model_train.png") )

        # save keras model
        self._model.save( os.path.join(self.__model_dir, "model.h5") )

    def fit(self):
        early_stop = EarlyStopping(monitor='val_loss', min_delta=self.__patience[0], 
                               patience=self.__patience[1], verbose=0)

        time0 = time.time()
        self._history = self._model.fit_generator(
                        self._generators["training_generator"], 
            epochs=self.__epochs,
            steps_per_epoch  = self._generators["steps_per_epoch_train"], 
            validation_data  = self._generators["validation_generator"], 
            validation_steps = self._generators["steps_per_epoch_val"],
            callbacks=[self._train_monitor, early_stop],
            verbose=self._config["verbose"],
            )
        self.__training_time = time.time() - time0

    def evaluate_loss(self):
        '''
        evaluate the model computing the score with 
        * the loss on the training set (the lower, the better. [0,1], usually ~0.6)
        * the difference between the loss on the training and validation dataset (the smaller, the better. )
        * the -log() of the pvalue of the kolmogorov test, if pval < 0.05 (better if non-existing)
        * the auc (the higher the better)
        we want to minimize the score
        '''
        logging.debug("Start Training")
        self.fit()
        logging.debug("Saving training metadata")
        self.save_metadata()
        ## we want a score to _minimize_
        ## the evaluation is based on the value of the loss
        score_loss  = self._train_monitor.val_losses[-1] 
        ## then we want to increase the score if the model overfits
        ## we add the difference between validation and training loss
        score_loss_ot = abs(self._train_monitor.losses[-1] - self._train_monitor.val_losses[-1] )
        ## we want to penalize the models for which the kolmogorov test fails
        ## if the test fail badly, the pval is for example 10*-15 or smaller: we add the -log, in this case +15
        ## then, we reduce this factor not to train _only_ on the kolmogorov test, but to give some importance to other factors
        score_ktest = 0.
        ## if self._train_monitor.kstest_sig[-1] < 0.05 or self._train_monitor.kstest_bkg[-1] < 0.05:
        #    score_ktest =  - 0.2 * np.log(min(self._train_monitor.kstest_sig[-1],self._train_monitor.kstest_bkg[-1])) 
        ## we want also to consider the auc. We want to encourage model with high auc
        ## we add (1-auc), with a factor to increase its importance
        score_auc    = - 1. / (1 - self._train_monitor.auc_test[-1])
        # as with loss, penalize if overtraining affects the auc
        score_auc_ot = abs(self._train_monitor.auc_train[-1] - self._train_monitor.auc_test[-1])
        logging.info(" - loss: "    + str(score_loss))
        logging.info(" - loss_ot: " + str(score_loss_ot))
        logging.info(" - ktest: "   + str(score_ktest))
        logging.info(" - auc: "     + str(score_auc))
        logging.info(" - auc_ot: "  + str(score_auc_ot))
        result = score_loss + score_loss_ot + score_ktest + score_auc + score_auc_ot
        logging.info("Result:  {}".format(result))
        return result

###############################################################################

def test_vbsdnn_model(config):
    _vbs_dnn = VbsDnn(
        config=config,
        input_dim=10,
        batch_size=1024,
        n_layers=1,
        n_nodes=30,
        dropout=0.4,
        batch_norm=True,
        epochs=3,
        test_ratio = 0.1,
        val_ratio = 0.2,
        patience = (0.001, 10)
    )
    evaluation = _vbs_dnn.evaluate_loss()
    return evaluation

def evaluate_vbsdnn_model(config, fixed_params, x):
    batch_size =int(x[:,0])
    n_layers = int(x[:,1])
    n_nodes = int(x[:,2])
    dropout = float(x[:,3])
    batch_norm = bool(x[:,4])
    input_dim = int(x[:,5])

    if n_nodes < input_dim:
        n_nodes = input_dim

    epochs     =fixed_params["epochs"]
    test_ratio  =fixed_params["test_ratio"]
    val_ratio  =fixed_params["val_ratio"]
    patience = fixed_params["patience"]

    logging.info(f"> L:{n_layers} , N:{n_nodes}, BS:{batch_size}, D:{dropout:.2f}, BN:{batch_norm}, I:{input_dim}")
    _vbs_dnn = VbsDnn(
        config=config,
        input_dim = input_dim,
        batch_size=batch_size,
        n_layers=n_layers,
        n_nodes=n_nodes,
        dropout=dropout,
        batch_norm=batch_norm,
        test_ratio = test_ratio,
        val_ratio = val_ratio,
        epochs=epochs, 
        patience = patience
    )
    evaluation = _vbs_dnn.evaluate_loss()
    return evaluation, _vbs_dnn
