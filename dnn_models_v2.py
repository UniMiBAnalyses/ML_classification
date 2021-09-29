
import tensorflow as tf

#from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout , LeakyReLU

lerelu = LeakyReLU(alpha=0.3)

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

    if model_tag == "3l_triangle128_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_triangle128_l2_batchnorm_relu_v2":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_64n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
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

    if model_tag == "3l_100n_l2_relu_batchnorm":
        model.add(Dense(100, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_triangle_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "3l_triangle256_l2_batchnorm_relu":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.03)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
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

    if model_tag == "4l_64n_l2_relu_batchnorm":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "4l_64_32n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_32n_l2_batchnorm_relu":
        model.add(Dense(32, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_64n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_128n_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_128n_l2_batchnorm_dropout_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.05)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.05)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.01))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_128n_l2_batchnorm_dropout_relu_v2":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.05)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(128,))
        model.add(Activation('relu'))
        model.add(Dropout(0.008))
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "4l_triangle_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "4l_triangle128_64_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
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

    if model_tag == "4l_256n_l2_batchnorm_relu":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_triangle_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    
    if model_tag == "5l_triangle_l2_batchnorm_relu_v2":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_triangle_l2_batchnorm_relu_v3":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_triangle_l2_batchnorm_relu_v4":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.03)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
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

    if model_tag == "5l_64n_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
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
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_64_32n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_64n_l2_batchnorm_selu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("selu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("selu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("selu")) 
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("selu")) 
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("selu")) 
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_64n_l2_relu_batchnorm_small":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64)) #kernel_regularizer=regularizers.l2(0.01)
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_64n_l2_relu_batchnorm":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64)) #kernel_regularizer=regularizers.l2(0.01)
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_32n_l2_batchnorm_relu":
        model.add(Dense(32, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    # if model_tag == "5l_64n_l2_mish_batchnorm":
    #     model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
    #     model.add(Activation(mish))
    #     model.add(BatchNormalization())
    #     model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
    #     model.add(Activation(mish))
    #     model.add(BatchNormalization())
    #     model.add(Dense(64)) #kernel_regularizer=regularizers.l2(0.01)
    #     model.add(Activation(mish))
    #     model.add(BatchNormalization())
    #     model.add(Dense(64))
    #     model.add(Activation(mish))
    #     model.add(BatchNormalization())
    #     model.add(Dense(64))
    #     model.add(Activation(mish))
    #     model.add(BatchNormalization())
    #     model.add(Dense(1, activation="sigmoid"))
    #     return model

    if model_tag == "5l_64n_l2_swish_batchnorm":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation(tf.nn.swish))
        model.add(BatchNormalization())
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation(tf.nn.swish))
        model.add(BatchNormalization())
        model.add(Dense(64)) #kernel_regularizer=regularizers.l2(0.01)
        model.add(Activation(tf.nn.swish))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation(tf.nn.swish))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation(tf.nn.swish))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "5l_64n_l2_batchnorm_swish":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.swish))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.swish))
        model.add(Dense(64)) #kernel_regularizer=regularizers.l2(0.01)
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.swish))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.swish))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.swish))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "5l_128_64n_dropout01_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
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

    if model_tag == "6l_triangle_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_triangle_l2_batchnorm_relu_v2":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.005)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_triangle_l2_batchnorm_relu_v3":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
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

    if model_tag == "6l_128n_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model



    if model_tag == "6l_128n_l2_relu_batchnorm":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_256n_l2_dropout_relu":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(Dropout(0.1))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(Dropout(0.1))
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(Dropout(0.1))
        model.add(Dense(128))
        model.add(lerelu)
        model.add(Dropout(0.1))
        model.add(Dense(64))
        model.add(lerelu)
        model.add(Dropout(0.1))
        model.add(Dense(64))
        model.add(lerelu)
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_256n_l2_batchnorm_relu":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(BatchNormalization())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(BatchNormalization())
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(lerelu)
        model.add(BatchNormalization())
        model.add(Dense(128))
        model.add(lerelu)
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(lerelu)
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(lerelu)
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "6l_256n_l2_batchnorm_tanh":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(Dense(1, activation="sigmoid"))
        return model

    
    if model_tag == "8l_64n_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dense(1, activation="sigmoid"))
        return model