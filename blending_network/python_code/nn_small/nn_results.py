import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
import pickle

np.random.seed(12345)
df = pd.read_csv('E2E_nn_data.csv')

dfin = df.iloc[:, 0:3]
dfout = df.iloc[:, 4:]

n_train = np.asarray(np.array([5, 10, 20, 50, 75, 100, 200, 400, 800, 1600, 3200, 4000.,])*1.2, dtype = 'int')
# n_train = [1000]
pred_error = np.empty(shape=[len(n_train), 10])
c = [6, 16, 10, 3, 13, 7]
d = [9, 15, 6, 12]
f_bar = [0, 200, 0, 50, 0, 350]

def pooling_objective(x):
    temp_x = [np.sum([x[i] for i in range(j, j+13, 4)]) for j in range(6,10)]
    return np.dot(c, x[0:6]) + (1/1000)*np.dot(np.square(x[0:6] - f_bar), c) - np.dot(temp_x, d)

for i in range(len(n_train)):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)   #top left
    ax2 = fig.add_subplot(212)   #top right
    for n_instance in range(10):
        # train our model
        train_idx = np.random.choice(dfin.shape[0], n_train[i], replace=False)
        dfin_train = dfin.iloc[train_idx]
        dfout_train = dfout.iloc[train_idx]

        dfin_mean = dfin_train.mean()
        dfin_std = dfin_train.std()
        dfout_mean = dfout_train.mean()
        dfout_std = dfout_train.std()

        dfin_train = (dfin_train - dfin_mean).divide(dfin_std)
        dfout_train = (dfout_train - dfout_mean).divide(dfout_std)

        # create our Keras Sequential model
        keras.backend.clear_session()
        nn = Sequential(name='relu_2_11')
        nn.add(Dense(units=11, input_dim=3, activation='relu'))
        nn.add(Dense(units=11, activation='relu'))
        # nn.add(Dense(units=16, activation='relu'))
        # nn.add(Dense(units=40, activation='relu'))
        # nn.add(Dense(units=40, activation='relu'))
        # nn.add(Dense(units=40, activation='relu'))
        # nn.add(Dense(units=40, activation='relu'))
        nn.add(Dense(units=22))
 
        nn.compile(optimizer=Adam(), loss='mse', metrics=[keras.metrics.MeanSquaredError()])
        x = dfin_train.values
        y = dfout_train.values
        
        # number of parameters
        n_params = (np.sum([np.prod(v.shape) for v in nn.trainable_variables]))

        earlystopping = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',
                                                        mode='min', patience=50,
                                                        restore_best_weights=True)
        history = nn.fit(x, y, epochs=5000, validation_split = 0.2, verbose = False, 
                         callbacks=[earlystopping])

        ax1.plot(history.history['mean_squared_error'])
        ax2.plot(history.history['val_mean_squared_error'])
        ax1.set_ylabel('mse loss')
        ax1.set_xlabel('epoch')
        ax2.set_ylabel('validation mse loss')
        ax2.set_xlabel('epoch')

        n_test = 100
        test_idx = np.random.choice(dfin.shape[0], n_test, replace=False)
        dfin_test = (dfin.iloc[test_idx] - dfin_mean).divide(dfin_std)
        dfout_test = (dfout.iloc[test_idx] - dfout_mean).divide(dfout_std)
        x_test = dfin_test.values
        y_test = dfout_test.values
        y_test = np.reshape(y_test, (n_test, 22))
        predictions = nn.predict(x_test)
        predictions = np.reshape(predictions, (n_test, 22))
        
        y_test_unscaled = np.empty(shape=y_test.shape)
        predictions_unscaled = np.empty(shape =predictions.shape)
        true_obj = np.empty(shape=(n_test,))
        pred_obj = np.empty(shape=(n_test,))
        for v in range(n_test):
            y_test_unscaled[v, :] = np.add(np.multiply(y_test[v, :], dfout_std), dfout_mean)
            predictions_unscaled[v, :] = np.add(np.multiply(predictions[v, :], dfout_std), dfout_mean)
            true_obj[v] = pooling_objective(y_test_unscaled[v, :])
            pred_obj[v] = pooling_objective(predictions_unscaled[v, :])
        pred_error[i, n_instance] = np.linalg.norm(true_obj - pred_obj, 1)

    plt.savefig('nn_small/loss_' + str(n_train[i]) + '_' + str(n_params) + '.png')
    plt.close('all')

with open('nn_small/E2E_' + str(n_params) + '.pickle', 'wb') as file:
    pickle.dump(pred_error, file)
file.close()