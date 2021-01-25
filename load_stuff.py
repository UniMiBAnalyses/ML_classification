import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_training_arrays(config):

    config_base_dir = os.path.join(config["base_dir"], config["plot_config"])

    # create the model directory
    model_dir   = os.path.join(config_base_dir, config["cut"] , "models",  config["model_version"])
    os.makedirs(model_dir, exist_ok=True)

    import yaml
    model_config_file = open(model_dir + "/model_config.yml", "r")
    model_config = yaml.safe_load(model_config_file)

    for key in ["samples_version", "cols"]:
        config[key] = model_config[key]

    # load numpy
    samples_dir = os.path.join(config_base_dir, config["cut"] , "samples", config["samples_version"])
    import pickle
    signal = pickle.load(open(os.path.join(samples_dir, "for_training/signal_balanced.pkl"),     "rb"))
    bkg    = pickle.load(open(os.path.join(samples_dir, "for_training/background_balanced.pkl"), "rb"))

    X_sig = signal[config["cols"]].values
    X_bkg = bkg[config["cols"]].values
    Y_sig = np.ones(len(X_sig))
    Y_bkg = np.zeros(len(X_bkg))
    W_sig = (signal["weight_norm"]).values
    W_bkg = (bkg["weight_norm"]).values
    Wnn_sig = (signal["weight"]).values
    Wnn_bkg = (bkg["weight"]).values

    