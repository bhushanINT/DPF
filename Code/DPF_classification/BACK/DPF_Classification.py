#https://keras.io/examples/timeseries/timeseries_transformer_classification/

#https://github.com/mounalab/Multivariate-time-series-forecasting-keras

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from xgboost import XGBClassifier
from scipy.spatial import distance
import numpy as np
import os

from data import load_train_data, load_test_data

project_name = 'DPF_class_Trans'

n_classes = 2

MODEL_architecture = 'xgboost' # 'Trans', 'lstm', 'cnn', xgboost

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
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def build_lstm_model(
    input_shape_in,
    lstm_blocks,
    dropout=0,
):
    model = keras.Sequential()
    model.add(LSTM(lstm_blocks,  input_shape = input_shape_in))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))

    return model

def build_cnn_model(input_shape, nb_classes):

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation="softmax")(gap)

    return keras.Model(inputs=input_layer, outputs=output_layer)

def train():

    x_train, y_train = load_train_data()

    x_train = np.swapaxes(x_train,1,2)

    n_classes = len(np.unique(y_train))

    print(n_classes)

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

        model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=1e-4),metrics=["sparse_categorical_accuracy"])

    elif MODEL_architecture == 'lstm':
        ############ LSTM ############### https://github.com/abulbasar/neural-networks/blob/master/Keras%20-%20Multivariate%20time%20series%20classification%20using%20LSTM.ipynb
        model = build_lstm_model(input_shape, lstm_blocks=100, dropout=0.5)
        model.compile(loss="binary_crossentropy", metrics=[keras.metrics.binary_accuracy], optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    elif MODEL_architecture == 'cnn':
        ############ CNN ############### https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
        model = build_cnn_model(input_shape, n_classes)
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["sparse_categorical_accuracy"])

    elif MODEL_architecture == 'xgboost':
        model = XGBClassifier()
        print(x_train.shape)
        x_train = x_train[:,:,1] # ONLY TWO
        print(x_train.shape)
        #exit(0)
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]* x_train.shape[2]))
        for i in range(0,len(y_train)):
            print('MAX----------',np.mean(x_train[i,:]),'---',y_train[i])
        exit(0)
        model.fit(x_train, y_train)
        x_test, y_test = load_test_data()
        x_test = np.swapaxes(x_test,1,2)
        print(x_test.shape)
        x_test = x_test[:,:,1]# ONLY TWO
        print(x_test.shape)
        #exit(0)
        #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]* x_test.shape[2]))
        y_test[y_test == -1] = 0
        y_pred = model.predict(x_test)
        print(y_test)
        print(y_pred)
        print("Prediction Accuracy is-----",100*((y_test == y_pred).sum()/len(y_pred)),'%')
        for i in range(0,len(y_test)):
            print('MAX----------',np.mean(x_test[i,:]),'---',y_test[i])
        exit(0)
        
    
    model.summary()

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

    y_test[y_test == -1] = 0
    print(y_test.shape)
   
    model = load_model(os.path.join(model_dir, project_name + '.h5'))
 
    y_pred = model.predict(x_test, batch_size=1, verbose=1)
    print(y_pred)
    #exit(0)

    if MODEL_architecture == 'Trans':
        print(np.array((y_pred[:,0]>y_pred[:,1]),dtype=int))
        y_pred_bin = np.array((y_pred[:,0]>y_pred[:,1]),dtype=int)
    else:
        y_pred_bin = np.array((y_pred> 0.5),dtype=int)
        y_pred_bin = y_pred_bin[:,0]

    print(y_test)
    print(y_pred_bin)

    Pred_Accu = 100*((y_test == y_pred_bin).sum()/len(y_pred_bin))
    print("Prediction Accuracy is-----",Pred_Accu,'%')


if __name__ == '__main__':
    train()
    #predict()

