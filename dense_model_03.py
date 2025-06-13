#Single dense model with feedforward NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, r2_score
from numpy import loadtxt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shap

# Load and preprocess dataset
dataset = loadtxt('../jan_1_train_only_precip.csv', delimiter=',', skiprows=1)
X = dataset[:, :14]
Y = dataset[:, 14].reshape(-1, 1)
X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

input_scaler = MinMaxScaler()
output_scaler = StandardScaler()
x_train = input_scaler.fit_transform(x_train)
x_test = input_scaler.transform(x_test)
y_train = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

batch_size = 64
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_scaled)).batch(batch_size)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)

# Model definition: hybrid of deep+normalized
model = Sequential()
model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005), input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
mc = ModelCheckpoint('jan_1_train_only_precip.keras', monitor='val_loss', mode='min',
                     save_best_only=True, verbose=1)

history = model.fit(train_data, validation_data=val_data, epochs=5000,
                    verbose=1, callbacks=[es, mc])

train_mse = model.evaluate(x_train, y_train, verbose=0)
test_mse = model.evaluate(x_test, y_test_scaled, verbose=0)
print(f'Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

predictions = model.predict(x_test)
pred_rescaled = output_scaler.inverse_transform(predictions)
y_test_rescaled = output_scaler.inverse_transform(y_test_scaled)

np.savetxt("jan_1_only_precip_x_predictions.csv", pred_rescaled, delimiter=",")
np.savetxt("jan_1_only_precip_y_original.csv", y_test_rescaled, delimiter=",")

mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
r2 = r2_score(y_test_rescaled, pred_rescaled)
print(f"Test MAE: {mae:.4f}, R² Score: {r2:.4f}")

# Training loss plot
plt.figure()
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig('jan_1_only_precip_training_loss.pdf', format='pdf')
plt.show()

# Predictions vs Ground Truth
plt.figure()
plt.scatter(y_test_rescaled, pred_rescaled, alpha=0.5)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],
         [y_test_rescaled.min(), y_test_rescaled.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted')
plt.grid(True)
plt.savefig('jan_1_only_precip_true_vs_predicted.pdf', format='pdf')
plt.show()

# Residuals plot
residuals = y_test_rescaled.flatten() - pred_rescaled.flatten()
plt.figure()
plt.scatter(pred_rescaled, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals Plot')
plt.grid(True)
plt.savefig('residuals_plot.pdf', format='pdf')
plt.show()

