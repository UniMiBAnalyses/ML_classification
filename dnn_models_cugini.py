
import tensorflow as tf

#from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout , LeakyReLU

lerelu = LeakyReLU(alpha=0.3)

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    

    if model_tag == "3l":
        model.add(Dense(20, input_dim=input_dim, activation="relu"))
        model.add(Dense(20, activation="relu"))
        model.add(Dense(20, activation="relu"))        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "1l_32":
        model.add(Dense(32, input_dim=input_dim, activation="relu"))   
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "1l_64":
        model.add(Dense(64, input_dim=input_dim, activation="relu"))   
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_64":
        model.add(Dense(64, input_dim=input_dim, activation="relu"))   
        model.add(Dense(64, input_dim=input_dim, activation="relu"))   
        model.add(Dense(1, activation="sigmoid"))
        return model

