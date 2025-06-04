#Optuna ensemble
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import loadtxt

# Set full determinism
random.seed(0)
np.random.seed(0)
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

# Load and preprocess dataset
dataset = loadtxt('../global_sw_train.csv', delimiter=',', skiprows=1)
X = dataset[:, :7]
Y = dataset[:, 7].reshape(-1, 1)
X, Y = shuffle(X, Y, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

input_scaler = MinMaxScaler()
output_scaler = StandardScaler()
x_train = input_scaler.fit_transform(x_train)
x_test = input_scaler.transform(x_test)
y_train = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

x_train_noisy = x_train + np.random.normal(0, 0.01, x_train.shape)

batch_size = 64
train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_scaled)).batch(batch_size)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)

# Optuna hyperparameter tuning
def build_model_optuna(trial):
    n_units1 = trial.suggest_int("units1", 128, 512, step=64)
    n_units2 = trial.suggest_int("units2", 64, 256, step=64)
    n_units3 = trial.suggest_int("units3", 32, 128, step=32)
    n_units4 = trial.suggest_int("units4", 16, 64, step=16)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
    l2_reg = trial.suggest_float("l2", 1e-5, 1e-3, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(n_units1, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(n_units2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(n_units3, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(n_units4, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def objective(trial):
    model = build_model_optuna(trial)
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    history = model.fit(train_data, validation_data=val_data, epochs=100, callbacks=[es, lr_sched], verbose=0)
    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)

# Save best parameters to reuse for ensemble
best_params = study.best_trial.params
np.save("optuna_best_params.npy", best_params)

print("Optuna hyperparameter tuning completed.")

# === Retrain ensemble using best Optuna parameters ===
best_params = np.load("optuna_best_params.npy", allow_pickle=True).item()

# Dataset already preprocessed above (x_train, y_train, etc.)
train_data = tf.data.Dataset.from_tensor_slices((x_train_noisy, y_train)).batch(batch_size).with_options(options)
val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_scaled)).batch(batch_size).with_options(options)


def build_model_from_best(params):
    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(params['units1'], kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(params['l2']))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout'])(x)

    x = Dense(params['units2'], kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(params['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout'])(x)

    x = Dense(params['units3'], kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(params['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout'])(x)

    x = Dense(params['units4'], kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(params['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout'])(x)

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')
    return model

print("Retraining ensemble with best Optuna hyperparameters...")
models = []
histories = []
preds = []
val_losses = []
seeds = [1, 42, 101, 202, 303, 404, 505]

for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    model = build_model_from_best(best_params)
    es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    mc = ModelCheckpoint(f'ensemble_optuna_model_seed{seed}.keras', monitor='val_loss', mode='min', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)

    history = model.fit(train_data, validation_data=val_data, epochs=1000, callbacks=[es, mc, lr], verbose=0)
    val_loss = min(history.history['val_loss'])
    models.append(model)
    histories.append(history)
    val_losses.append(val_loss)
    preds.append(model.predict(x_test))

# Weighted ensemble prediction with Optuna-tuned models
inv_losses = 1 / np.array(val_losses)
weights = inv_losses / np.sum(inv_losses)
all_preds = np.array(preds)
ensemble_preds = np.average(all_preds, axis=0, weights=weights)
ensemble_std = np.std(all_preds, axis=0)

ensemble_preds_rescaled = output_scaler.inverse_transform(ensemble_preds)
ensemble_std_rescaled = ensemble_std * output_scaler.scale_
y_test_rescaled = output_scaler.inverse_transform(y_test_scaled)

np.savetxt("ensemble_optuna_predictions.csv", ensemble_preds_rescaled, delimiter=",")
np.savetxt("ensemble_optuna_uncertainty.csv", ensemble_std_rescaled, delimiter=",")
np.savetxt("ensemble_optuna_true_values.csv", y_test_rescaled, delimiter=",")

mae = mean_absolute_error(y_test_rescaled, ensemble_preds_rescaled)
r2 = r2_score(y_test_rescaled, ensemble_preds_rescaled)
print(f"Optuna Ensemble MAE: {mae:.4f}, R² Score: {r2:.4f}")

# Plot validation losses
plt.figure(figsize=(12, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['val_loss'], label=f'Model {i+1} (Seed {seeds[i]})')
plt.title('Validation Loss (Optuna Ensemble)')
plt.xlabel('Epoch')
plt.ylabel('Validation MSE')
plt.legend()
plt.grid(True)
plt.savefig('optuna_ensemble_val_loss_comparison.pdf')
plt.show()

# True vs Predicted with uncertainty
plt.figure()
plt.errorbar(y_test_rescaled.flatten(), ensemble_preds_rescaled.flatten(), yerr=ensemble_std_rescaled.flatten(), fmt='o', alpha=0.4)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values with Uncertainty')
plt.title('True vs. Optuna Ensemble Predicted')
plt.grid(True)
plt.savefig('optuna_ensemble_true_vs_predicted.pdf')
plt.show()

# Residuals plot
residuals = y_test_rescaled.flatten() - ensemble_preds_rescaled.flatten()
plt.figure()
plt.scatter(ensemble_preds_rescaled, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Optuna Ensemble Residuals')
plt.grid(True)
plt.savefig('optuna_ensemble_residuals_plot.pdf')
plt.show()