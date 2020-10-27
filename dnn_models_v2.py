
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, BatchNormalization, Dropout 

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    

    if model_tag == "3l_100n_nodropout_nobatch_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))        
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_200n_dropout01_nobatch_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(200, activation="relu")) 
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_200n_dropout03_nobatch_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(200, activation="relu")) 
        model.add(Dense(1, activation="sigmoid"))
        return model


    if model_tag == "5l_300n_nodropout_nobatch_relu":
        model.add(Dense(300, input_dim=input_dim, activation="relu"))
        for i in range(4):
            model.add(Dense(300, activation="relu"))   
        model.add(Dense(1, activation="sigmoid"))
        return model
       
    if model_tag == "5l_50n_dropout01_nobatch_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        for i in range(4):
            model.add(Dropout(0.1))
            model.add(Dense(50, activation="relu"))   
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_150n_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_100n_l2_batchnorm_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_50n_dropout005_l2_relu":
        model.add(Dense(80, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1))
        model.add(Dense(50,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.05))
        model.add(Dense(50,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(50,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_100n_dropout005_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_100n_dropout005_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_100n_dropout005_nol2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_150n_dropout005_l2005_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.005)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.005)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.005)))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_200_50n_l2_batchnorm_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(50,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(50,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_56n_l2_batchnorm_relu":
        model.add(Dense(56, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(56,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(56,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(56,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(56,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(56,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_64n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model