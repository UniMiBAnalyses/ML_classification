import plot_loss

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout

import numpy as np

import os
import datetime

import logging
logging.basicConfig(level=logging.INFO)

class VbsDnn():
    def __init__ (self,
        config,
        input_dim, 
        dense_outs=(50,80,50),
        dense_drop=(0.3,0.3,0.3),
        batch_size=1024,
        epochs=50,
        test_ratio=0.2,
        val_ratio=0.0
        ):

        self.__config = config
        self.__input_dim = input_dim
        self.__test_ratio = test_ratio
        self.__val_ratio = val_ratio
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__model_dir = None
        self.dense_outs = dense_outs
        self.dense_drop = dense_drop

        self.__data_split, self.__generators = self.data_loader()

        self.__model = self.get_model()

        self.__train_monitor = plot_loss.PlotLosses(self.__model, self.__data_split, batch_mode=True)

        self.__history = None

    def data_loader(self):
        '''
        This can be used also to evaluate the input variables with AVOVA F-test
        '''
        ## read data from files
        logging.info("Loading data")
        config_base_dir = os.path.join(self.__config["base_dir"], self.__config["plot_config"])

        # create the model directory
        utc_seconds = str(datetime.datetime.now().timestamp()).split(".")[0]
        logging.info(utc_seconds)
        self.__model_dir   = os.path.join(config_base_dir, self.__config["cut"] , "optimization", utc_seconds)
        os.makedirs(self.__model_dir, exist_ok=True)

        # load numpy
        samples_dir = os.path.join(config_base_dir, self.__config["cut"] , "samples", self.__config["samples_version"])
        import pickle
        signal = pickle.load(open(os.path.join(samples_dir, "for_training/signal_balanced.pkl"),     "rb"))
        bkg    = pickle.load(open(os.path.join(samples_dir, "for_training/background_balanced.pkl"), "rb"))

        ## create numpy arrays
        X_sig = signal[self.__config["cols"]].values
        X_bkg = bkg[self.__config["cols"]].values
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
        X_train, X_test, y_train, y_test, W_train, W_test , Wnn_train, Wnn_test = train_test_split(X_scaled, Y,  W, Wnn, test_size=self.__test_ratio)
        # if self.__val_ratio != 0.0:
        #     X_val, X_test, y_val, y_test, W_val, W_test = train_test_split(X_test,   y_test, W_test, test_size=config["val_size"]) ## test != val

        data_split = {
            "X_train": X_train,
            "X_test" : X_test, ## test != val
            "y_train": y_train,
            "y_test" : y_test, ## test != val
            "W_train": W_train,
            "W_test" : W_test, ## test != val
            "Wnn_train": Wnn_train,
            "Wnn_test" : Wnn_test, ## test != val
        }

        ## Oversampling
        training_generator,   steps_per_epoch_train = balanced_batch_generator(X_train, y_train, W_train, batch_size=self.__batch_size, sampler=RandomOverSampler())
        #validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_val,   y_val,   W_val,   batch_size=self.__batch_size, sampler=RandomOverSampler()) ## test != val
        validation_generator, steps_per_epoch_val   = balanced_batch_generator(X_test,  y_test,  W_test,   batch_size=self.__batch_size, sampler=RandomOverSampler()) ## test == val

        generators = {
            "training_generator": training_generator,
            "steps_per_epoch_train": steps_per_epoch_train,
            "validation_generator": validation_generator,
            "steps_per_epoch_val": steps_per_epoch_val,
        }

        return data_split, generators

    def get_model(self):
        '''

        '''
        logging.info("Creating model")
        model = Sequential()
        for idx,dense_props in enumerate(zip(self.dense_outs, self.dense_drop)):
            outs = dense_props[0]
            dropout = dense_props[1]
            if idx==0:
                model.add(Dense(outs, input_dim=self.__input_dim))
            else:
                model.add(Dense(outs))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        return model

    def save_metadata(self):
        ## SAVE THE MODEL, ITS METADATA AND TRAINING INFORMATIONS

        ## FIXME
        ## Retrieve the model metadata that should be saved
        # # dump the variables list
        # import yaml
        # varfile = os.path.join(self.__model_dir, "variables.yml")
        # if os.path.isfile(varfile):
        #     print("ACHTUNG! variables file already existing: old file renamed with '_old'")
        #     os.rename(varfile, varfile[:-4] + "_old.yml")
        # with open(varfile, "w") as out_var_file:
        #     out_var_file.write(yaml.dump(config["cols"]))
            
        # # dump the config
        # model_config_file = os.path.join(self.__model_dir, "model_config.yml")
        # if os.path.isfile(model_config_file):
        #     print("ACHTUNG! model_config_file file already existing: old file renamed with '_old'")
        #     os.rename(model_config_file, model_config_file[:-4] + "_old.yml")
        # with open(model_config_file, "w") as out_var_file:
        #     out_var_file.write(yaml.dump(config))  

        # save figure with training summary
        self.__train_monitor.save_figure( os.path.join(self.__model_dir, "model_train.png") )

        # save keras model
        self.__model.save( os.path.join(self.__model_dir, "model.h5") )

    def fit(self):
        self.__history = self.__model.fit_generator(
            self.__generators["training_generator"], 
            epochs=self.__epochs,
            steps_per_epoch  = self.__generators["steps_per_epoch_train"], 
            validation_data  = self.__generators["validation_generator"], 
            validation_steps = self.__generators["steps_per_epoch_val"],
            callbacks=[self.__train_monitor],
            verbose=0,
            )

    def evaluate_loss(self):
        '''
        evaluate the model based on the loss on the training set
        '''
        logging.info("Start Training")
        self.fit()
        evaluation = self.__model.evaluate_generator(self.__generators["validation_generator"], steps=1000)
        logging.info("Saving training metadata")
        self.save_metadata()
        return evaluation ## LOSS

    # def evaluate_auc():
    #     '''
    #     evaluate the model based on the integral of the roc curve on the training set
    #     '''
    #     self.fit()

def test_vbsdnn_model():
    config = {
        "base_dir":        "/eos/home-d/dmapelli/public/latino/",
        "plot_config":     "Full2018v6s5",
        "cut":             "boos_sig_mjjincl",
        "samples_version": "v10",
        "cols": ['mjj_vbs', 'vbs_0_pt', 'vbs_1_pt', 'deltaeta_vbs', 'deltaphi_vbs', 
            'mjj_vjet', 'vjet_0_pt', 'vjet_1_pt', 'vjet_0_eta', 'vjet_1_eta', 
            'Lepton_pt', 'Lepton_eta', 'Lepton_flavour', 
            'PuppiMET', 
            'Zvjets_0', 'Zlep', 
            'Asym_vbs', 'Asym_vjet', 'A_ww', 
            'Mtw_lep', 'w_lep_pt', 'Mww', 'R_ww', 'R_mw', 
            'Centr_vbs', 'Centr_ww'
            ]
    }
    input_dim = len(config["cols"])
    dense_outs=(50,80,50)
    dense_drop=(0.3,0.3,0.3)
    batch_size=1024
    epochs=3
    test_ratio=0.2
    val_ratio=0.0
    _vbs_dnn = VbsDnn(
        config=config,
        input_dim=input_dim,
        dense_outs=dense_outs,
        dense_drop=dense_drop,
        batch_size=batch_size,
        epochs=epochs,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )
    evaluation = _vbs_dnn.evaluate_loss()
    return evaluation

def evaluate_vbsdnn_model(config, fixed_params, x):
    test_ratio=float(x[:,0])
    batch_size =int(x[:,1])

    input_dim  = fixed_params["input_dim"]
    dense_drop =fixed_params["dense_drop"]
    # batch_size =fixed_params["batch_size"]
    dense_outs =fixed_params["dense_outs"]
    epochs     =fixed_params["epochs"]
    val_ratio  =fixed_params["val_ratio"]

    _vbs_dnn = VbsDnn(
        config=config,
        input_dim=input_dim,
        dense_outs=dense_outs,
        dense_drop=dense_drop,
        batch_size=batch_size,
        epochs=epochs,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )
    evaluation = _vbs_dnn.evaluate_loss()
    return evaluation
