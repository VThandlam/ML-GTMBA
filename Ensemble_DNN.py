#Ensemble DNN with 7 seeds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
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
import pandas as pd

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

def build_model():
    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

# Train ensemble of models
models = []
histories = []
preds = []
seeds = [1, 42, 101, 202, 303, 404, 505]
for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    model = build_model()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
    mc = ModelCheckpoint(f'ensemble_model_seed{seed}.keras', monitor='val_loss', mode='min', save_best_only=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
    history = model.fit(train_data, validation_data=val_data, epochs=1000, callbacks=[es, mc], verbose=0)
    models.append(model)
    histories.append(history)
    preds.append(model.predict(x_test))

# Plot individual loss curves
plt.figure(figsize=(12, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['val_loss'], label=f'Model {i+1} (Seed {seeds[i]})')
plt.title('Validation Loss Across Ensemble Members')
plt.xlabel('Epoch')
plt.ylabel('Validation MSE')
plt.legend()
plt.grid(True)
plt.savefig('ensemble_val_loss_comparison.pdf')
plt.show()

# Ensemble prediction and uncertainty
all_preds = np.array(preds)
ensemble_preds = np.mean(all_preds, axis=0)
ensemble_std = np.std(all_preds, axis=0)

ensemble_preds_rescaled = output_scaler.inverse_transform(ensemble_preds)
ensemble_std_rescaled = ensemble_std * output_scaler.scale_  # rescale std

# Save predictions and uncertainty
np.savetxt("ensemble_predictions.csv", ensemble_preds_rescaled, delimiter=",")
np.savetxt("ensemble_uncertainty.csv", ensemble_std_rescaled, delimiter=",")
y_test_rescaled = output_scaler.inverse_transform(y_test_scaled)
np.savetxt("ensemble_true_values.csv", y_test_rescaled, delimiter=",")

# Metrics
mae = mean_absolute_error(y_test_rescaled, ensemble_preds_rescaled)
r2 = r2_score(y_test_rescaled, ensemble_preds_rescaled)
print(f"Ensemble MAE: {mae:.4f}, R² Score: {r2:.4f}")

# True vs Predicted
plt.figure()
plt.errorbar(y_test_rescaled.flatten(), ensemble_preds_rescaled.flatten(), yerr=ensemble_std_rescaled.flatten(), fmt='o', alpha=0.4)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values with Uncertainty')
plt.title('True vs. Ensemble Predicted with Uncertainty')
plt.grid(True)
plt.savefig('ensemble_true_vs_predicted_uncertainty.pdf')
plt.show()

# Residuals
residuals = y_test_rescaled.flatten() - ensemble_preds_rescaled.flatten()
plt.figure()
plt.scatter(ensemble_preds_rescaled, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)
plt.savefig('ensemble_residuals_plot.pdf')
plt.show()
