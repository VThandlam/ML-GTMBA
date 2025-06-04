# mlp for regression with mse loss function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.utils import shuffle
import numpy as np
import tensorflow.keras
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf

#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():

# load the dataset
dataset = loadtxt('sw_training.csv', delimiter=',', skiprows=1)

# split into input (X) and output (y) variables
X = dataset[:,0:11]
Y = dataset[:,11].reshape(len(dataset[:,11]),1)

X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#Scalling the data
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()
x_train = input_scaler.fit_transform(x_train)
x_test = input_scaler.fit_transform(x_test)
y_train = output_scaler.fit_transform(y_train)
y_test = output_scaler.fit_transform(y_test)


# Wrap data in Dataset objects.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# The batch size must now be set on the Dataset objects.
batch_size = 32
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)

# define the keras model
model = Sequential()

# Model
model.add(Dense(600, input_dim = x_train.shape[1], activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the network :
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='mse', optimizer=opt)
model.summary()

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('sw_deeplearning.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# fit model
history = model.fit(train_data, validation_data=val_data, epochs=10000, verbose=1, callbacks=[es, mc])    

# evaluate the model
train_mse = model.evaluate(x_train, y_train, verbose=1)
test_mse = model.evaluate(x_test, y_test, verbose=1)
print('Train: %.2f, Test: %.2f' % (train_mse, test_mse))

# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('sw_dl_training_loss.pdf', format='pdf')
pyplot.show()

#make regress predictions with the model
predictions = model.predict(x_test)
rescaled= output_scaler.inverse_transform(predictions)
y_test = output_scaler.inverse_transform(y_test)
np.savetxt("sw_x_test_predictions.csv", rescaled, delimiter=",")
np.savetxt("sw_y_test.csv", y_test, delimiter=",")
print(y_test.shape, rescaled.shape)

