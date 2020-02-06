# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()

    if model_tag == "boost_5vars_v0":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))
        model.add(Dense(1, activation="sigmoid"))

        #model.add(Dropout(0.05))
        #model.add(Dense(50, activation="relu"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    if model_tag == "res_3depth_v0":
        model.add(Dense(100, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))

        model.add(Dense(80))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(1, activation="sigmoid"))

        #model.add(Dropout(0.05))
        #model.add(Dense(50, activation="relu"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    if model_tag == "res_3depth_v1":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))

        model.add(Dense(80))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(1, activation="sigmoid"))

        #model.add(Dropout(0.05))
        #model.add(Dense(50, activation="relu"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    if model_tag == "res_5depth_v0":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))

        model.add(Dense(80))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(80))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.03))

        model.add(Dense(1, activation="sigmoid"))

        #model.add(Dropout(0.05))
        #model.add(Dense(50, activation="relu"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # if id=="multi_small":
    #     model.add(Dense(units=100,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=100,activation="relu"))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=100,activation="relu"))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=100,activation="relu"))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_medium":
    #     model.add(Dense(units=300,input_dim=input_dim))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=200))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=100))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_big":
    #     model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_big_nodropout":
    #     model.add(Dense(units=500,input_dim=input_dim))
    #     model.add(LeakyReLU()) 
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_big_3":
    #     model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=400, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=300, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=200, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=100, activation="relu"))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_big_4":
    #     model.add(Dropout(0.1))
    #     model.add(Dense(units=600,input_dim=input_dim))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=600))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=600))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=600))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="multi_very_big":
    #     model.add(Dense(units=500,input_dim=input_dim))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=500))
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=5, activation='softmax'))

    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer="adam",
    #                   metrics=[metrics.categorical_accuracy])

    # elif id=="binary_small":
    #     model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
    #     model.add(Dense(units=200,activation="relu"))
    #     model.add(Dense(units=100,activation="relu"))
    #     model.add(Dense(units=50,activation="relu"))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="binary_dropout":
    #     model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(units=400,activation="relu"))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(units=300,activation="relu"))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(units=300,activation="relu"))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="binary_dropout_small":
    #     model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=200,activation="relu"))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(units=100,activation="relu"))
    #     model.add(Dropout(0.20))
    #     model.add(Dense(units=50,activation="relu"))
    #     model.add(Dropout(0.10))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])


    # elif id=="binary_dropout2":
    #     model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="binary_dropout3":
    #     model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=500,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=300,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="binary_dropout4":
    #     model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=300,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=300,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="binary_dropout5":
    #     model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=200,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=200,activation="relu"))
    #     model.add(Dropout(0.30))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="very_small_test":
    #     model.add(Dense(units=50,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.10))
    #     model.add(Dense(units=50,activation="relu"))
    #     model.add(Dropout(0.10))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    # elif id=="smallest":
    #     model.add(Dense(units=100,input_dim=input_dim, activation="relu"))
    #     model.add(Dropout(0.10))
    #     model.add(Dense(units=1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                 optimizer="adam",
    #                 metrics=[metrics.binary_accuracy])

    
    return model