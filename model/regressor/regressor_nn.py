from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# model for 'fair' dataset 
def create_model_fair():
    model = Sequential()
    # here in input argument you shoul fill with amount of features in dataset you are using
    model.add(Dense(144, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# model considering all features
def create_model_all():
    model = Sequential()
    # here in input argument you shoul fill with amount of features in dataset you are using
    model.add(Dense(144, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def loss_history_model(model, X, Y):
    early_stopping_monitor = EarlyStopping(patience=10)
    scalar = MinMaxScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    history = model.fit(X, Y, validation_split=0.3, epochs=200, batch_size=1, verbose=1, callbacks=[early_stopping_monitor])

    scores = model.evaluate(X, Y, verbose=1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return None
