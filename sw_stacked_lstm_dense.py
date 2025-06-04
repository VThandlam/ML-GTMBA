import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sw_training.csv', header=0)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Split data into input (predictor) and output (target) variables
X = scaled_data[:, :11]
y = scaled_data[:, 11]

# Reshape input data to be 3-dimensional
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Train-test split
split_index = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model with dense layers
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(1, 11), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the network
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt)

# Build the model
model.build(input_shape=(None, 1, 11))

model.summary()

# Define early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
model_save = ModelCheckpoint('sw_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train LSTM model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5000, batch_size=32, verbose=1, callbacks=[early_stop, model_save])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('sw_loss.png')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Save predictions to file
#np.savetxt('sw_y_test.csv', y_test, delimiter=',')
#np.savetxt('sw_y_pred.csv', y_pred, delimiter=',')

