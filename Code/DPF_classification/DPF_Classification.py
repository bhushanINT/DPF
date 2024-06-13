#https://keras.io/examples/timeseries/timeseries_transformer_classification/

#https://github.com/mounalab/Multivariate-time-series-forecasting-keras

#https://www.kaggle.com/code/yahyamomtaz/rul-prediction-using-lstm-for-aircraft-engine

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from xgboost import XGBClassifier
from scipy.spatial import distance
from matplotlib import pyplot as plt 
import numpy as np
import os

from data import load_train_data, load_test_data

project_name = 'Trans'

MODEL_architecture = 'Trans' # 'Trans', 'lstm', 'cnn', xgboost

model_dir = "D:/Work/Timeseries_models/Code/DPF_classification/Model/"


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    #inputs = keras.Input(shape=(None,input_shape[1]))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    #outputs = layers.Dense(1, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

def build_lstm_model(
    input_shape_in,
    lstm_blocks,
    dropout,
):
    model = keras.Sequential()
    model.add(LSTM(lstm_blocks, input_shape = input_shape_in, return_sequences=True, activation = "tanh"))
    #model.add(LSTM(lstm_blocks, input_shape = (None,input_shape_in[1]), return_sequences=True, activation = "tanh"))
    model.add(LSTM(64, activation = "tanh", return_sequences = True))
    model.add(LSTM(32, activation = "tanh"))
    #model.add(Dropout(dropout))
    model.add(Dense(96, activation = "relu"))
    #model.add(Dropout(dropout))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(1))

    return model



def train():

    x_train, y_train = load_train_data()

    x_train = np.swapaxes(x_train,1,2)

    print(x_train.shape)
    print(y_train.shape)
    #exit(0)

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]


    y_train[y_train == -1] = 0

    print(y_train.shape)

    input_shape = x_train.shape[1:]
    print(input_shape)

    if MODEL_architecture == 'Trans':
        ############ TRANSFORMER ############### https://keras.io/examples/timeseries/timeseries_transformer_classification/
        #model = build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=1, mlp_units=[32], mlp_dropout=0.4, dropout=0.25)
        model = build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
        #model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=1e-4),metrics=["sparse_categorical_accuracy"])
        #model.compile(loss = "mse", metrics=[keras.metrics.CosineSimilarity(axis=1)], optimizer=keras.optimizers.Adam(learning_rate=0.001))
        model.compile(loss = "mae", metrics=[keras.metrics.CosineSimilarity(axis=1)], optimizer=keras.optimizers.Adam(learning_rate=0.001))

    elif MODEL_architecture == 'lstm':
        ############ LSTM ############### https://github.com/abulbasar/neural-networks/blob/master/Keras%20-%20Multivariate%20time%20series%20classification%20using%20LSTM.ipynb
        model = build_lstm_model(input_shape, lstm_blocks=128, dropout=0.5)
        model.compile(loss = "mse", metrics=[keras.metrics.CosineSimilarity(axis=1)], optimizer=keras.optimizers.Adam(learning_rate=0.001))

    
    model.summary()
    exit(0)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True)

    csv_logger = CSVLogger(os.path.join(model_dir,  project_name + '.csv'), separator=',', append=False)

    model.fit(x_train, y_train, validation_split=0.2, epochs=200, batch_size=2, callbacks=[model_checkpoint, csv_logger])

def predict():

    x_test, y_test = load_test_data()
    x_test = np.swapaxes(x_test,1,2)

    print(x_test.shape)
    print(y_test.shape)
    #exit(0)

    idx = np.random.permutation(len(x_test))
    x_test = x_test[idx]
    y_test = y_test[idx]

    y_test[y_test == -1] = 0
    print(y_test.shape)
    #exit(0)
   
    model = load_model(os.path.join(model_dir, project_name + '.h5'))
 
    y_pred = model.predict(x_test, batch_size=1, verbose=1)

    print(y_pred)
    print(y_test)

    plt.plot(y_pred, label='Prediction')
    plt.plot(y_test, label='GT')
    plt.ylabel('Remaining life in duty cycle ')
    plt.xlabel('Test Case')
    plt.legend(["Prediction", "GT"])
    plt.show()

    plt.plot(y_pred, label='Prediction')
    plt.plot(y_test, label='GT')
    plt.ylabel('Remaining life in duty cycle ')
    plt.xlabel('Test Case')
    plt.legend(["Prediction", "GT"])
    plt.show()


if __name__ == '__main__':
    #train()
    predict()

