import dnn_plot_loss

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
logging.basicConfig(level=logging.DEBUG)


class VbsDnn():
    def __init__ (self,
        dataload_config = {}, 
        model_config    = {},
        ):
        self.__model_dir = None
        self._history = None
        self.__training = {}

        self.__model_config    = deepcopy(model_config)
        self.__dataload_config = deepcopy(dataload_config)
        self.config_validation()

        self._data_split, self._generators = self.data_loader()
        self._model = self.get_model()
        self._train_monitor = dnn_plot_loss.PlotLosses(self._model, self._data_split, batch_mode=True)


    def config_validation(self):
        model_config_keys = [
            "input_dim",
            "batch_size",
            "epochs",
            "n_layers",
            "n_nodes",
            "dropout",
            "batch_norm",
            "patience",
            "verbose"
        ]
        for key in model_config_keys:
            if key not in self.__model_config:
                raise Exception(f"model configuration is missing the key: {key}")

        dataload_config_keys = [
            "base_dir",
            "plot_config",
            "cut",
            "samples_version",
            "cols",
            "test_ratio",
            "val_ratio",
        ]
        for key in dataload_config_keys:
            if key not in self.__dataload_config:
                raise Exception(f"dataload configuration is missing the key: {key}")

    def data_loader(self):
        '''
        This can be used also to evaluate the input variables with AVOVA F-test
        '''
        ## read data from files
        logging.debug("Loading data")
        config_base_dir = os.path.join(self.__dataload_config["base_dir"], self.__dataload_config["plot_config"])

        # create the model directory
        utc_seconds = str(datetime.datetime.now().timestamp()).split(".")[0]
        logging.info(utc_seconds)
        self.__model_dir   = os.path.join(config_base_dir, self.__dataload_config["cut"] , "steps", utc_seconds)
        os.makedirs(self.__model_dir, exist_ok=True)

        # load numpy
        samples_dir = os.path.join(config_base_dir, self.__dataload_config["cut"] , "samples", self.__dataload_config["samples_version"])
        signal = pickle.load(open(os.path.join(samples_dir, "for_training/signal_balanced.pkl"),     "rb"))
        bkg    = pickle.load(open(os.path.join(samples_dir, "for_training/background_balanced.pkl"), "rb"))

        # Keep only the first "input-dim" columns
        self.__dataload_config["cols"] = self.__dataload_config["cols"][:self.__model_config["input_dim"]]
        logging.debug(self.__dataload_config["cols"])

        ## create numpy arrays
        X_sig = signal[self.__dataload_config["cols"]].values
        X_bkg = bkg[self.__dataload_config["cols"]].values
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
                                        test_size=self.__dataload_config["test_ratio"], random_state=42, stratify=Y)
        X_train, X_val,  y_train,  y_val, W_train, W_val,   Wnn_train, Wnn_val =  train_test_split(X_train, y_train, W_train, Wnn_train, 
                                        test_size=self.__dataload_config["val_ratio"], random_state=42,  stratify=y_train) 

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

        # for dataset_name, dataset in data_split.items():
        #     print(dataset_name + " " + str(float(dataset.nbytes) / (1024. * 1024.)))

        ## Oversampling
        training_generator,   steps_per_epoch_train = balanced_batch_generator(X_train, y_train, W_train, batch_size=self.__model_config["batch_size"], sampler=RandomOverSampler())
        #validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_val,   y_val,   W_val,   batch_size=self.__model_config["batch_size"], sampler=RandomOverSampler()) ## test != val
        validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_val,  y_val,  W_val,   batch_size=self.__model_config["batch_size"], sampler=RandomOverSampler()) ## test == val

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
        for ilay in range(self.__model_config["n_layers"]):
            if ilay==0:
                model.add(Dense(self.__model_config["n_nodes"], input_dim=self.__model_config["input_dim"]))
            else:
                model.add(Dense(self.__model_config["n_nodes"]))
            if self.__model_config["batch_norm"]:
                model.add(BatchNormalization())
            model.add(Activation("relu"))
            if self.__model_config["dropout"] > 0:
                model.add(Dropout(self.__model_config["dropout"]))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        return model

    def save_metadata(self):
        ## SAVE THE MODEL, ITS METADATA AND TRAINING INFORMATIONS

        metadata = {
            "data_load" : self.__dataload_config,
            "model"     : self.__model_config,
            "training"  : {},
        }

        # # Retrieve the model metadata that should be saved
        # # dump the variables list
        # for v in self.__dict__:
        #     if v.startswith("_"+self.__class__.__name__+"__"):
        #         self._config[v.split("__")[1]] = getattr(self, v)
            
        # dump the config
        model_config_file = os.path.join(self.__model_dir, "model_metadata.yml")
        if os.path.isfile(model_config_file):
            print("ACHTUNG! model_config_file file already existing: old file renamed with '_old'")
            os.rename(model_config_file, model_config_file[:-4] + "_old.yml")
        with open(model_config_file, "w") as out_var_file:
            out_var_file.write(yaml.dump(metadata))  

        # save figure with training summary
        self._train_monitor.save_figure( os.path.join(self.__model_dir, "model_train.png") )

        # save keras model
        self._model.save( os.path.join(self.__model_dir, "model.h5") )

    def fit(self):
        early_stop = EarlyStopping(monitor='val_loss', min_delta=self.__model_config["patience"][0], 
                               patience=self.__model_config["patience"][1], verbose=0)

        time0 = time.time()
        self._history = self._model.fit_generator(
                            self._generators["training_generator"], 
                            epochs=self.__model_config["epochs"],
                            steps_per_epoch  = self._generators["steps_per_epoch_train"], 
                            validation_data  = self._generators["validation_generator"], 
                            validation_steps = self._generators["steps_per_epoch_val"],
                            callbacks=[self._train_monitor, early_stop],
                            verbose=self.__model_config["verbose"],
                            use_multiprocessing=True,
                            workers=10,
                            )
        self.__training["training_time"] = time.time() - time0

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
        ## we want a score to _minimize_
        ## the evaluation is based on the value of the loss
        self.__training["loss"]  = self._train_monitor.val_losses[-1] 
        ## then we want to increase the score if the model overfits
        ## we add the difference between validation and training loss
        self.__training["loss_ot"] = abs(self._train_monitor.losses[-1] - self._train_monitor.val_losses[-1] )
        ## we want to penalize the models for which the kolmogorov test fails
        ## if the test fail badly, the pval is for example 10*-15 or smaller: we add the -log, in this case +15
        ## then, we reduce this factor not to train _only_ on the kolmogorov test, but to give some importance to other factors
        self.__training["ktest"] = 0.
        ## if self._train_monitor.kstest_sig[-1] < 0.05 or self._train_monitor.kstest_bkg[-1] < 0.05:
        #    self.__training["ktest"] =  - 0.2 * np.log(min(self._train_monitor.kstest_sig[-1],self._train_monitor.kstest_bkg[-1])) 
        ## we want also to consider the auc. We want to encourage model with high auc
        ## we add (1-auc), with a factor to increase its importance
        self.__training["auc"]    = - 1. / (1 - self._train_monitor.auc_test[-1])
        # as with loss, penalize if overtraining affects the auc
        self.__training["auc_ot"] = abs(self._train_monitor.auc_train[-1] - self._train_monitor.auc_test[-1])
        logging.info(" - loss: "    + str(self.__training["loss"]))
        logging.info(" - loss_ot: " + str(self.__training["loss_ot"]))
        logging.info(" - ktest: "   + str(self.__training["ktest"]))
        logging.info(" - auc: "     + str(self.__training["auc"]))
        logging.info(" - auc_ot: "  + str(self.__training["auc_ot"]))
        result = self.__training["loss"] + self.__training["loss_ot"] + self.__training["ktest"] + self.__training["auc"] + self.__training["auc_ot"]
        logging.info("Result:  {}".format(result))
        self.save_metadata()
        return result

###############################################################################

def evaluate_vbsdnn_model(dataload_config, model_config):
    _vbs_dnn = VbsDnn(
        dataload_config = dataload_config,
        model_config    = model_config,
    )
    evaluation = _vbs_dnn.evaluate_loss()
    return evaluation, _vbs_dnn
    