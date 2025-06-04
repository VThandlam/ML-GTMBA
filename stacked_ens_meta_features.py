#Stacked Ensemble with Quantile Features
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# ========== Setup ==========
os.makedirs("models", exist_ok=True)
os.makedirs("models_quantile", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# ========== Load and Prepare Data ==========
dataset = loadtxt('../global_sw_train.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:7]
Y = dataset[:, 7].reshape(-1, 1)

X, Y = shuffle(X, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_scaler = RobustScaler()
y_scaler = RobustScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

joblib.dump(x_scaler, "models/ensemble_input_scaler.pkl")
joblib.dump(y_scaler, "models/ensemble_output_scaler.pkl")

# ========== Train Base Models ==========
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(x_train_scaled, y_train_scaled.ravel())
joblib.dump(knn, "models/knn.pkl")
knn_pred = knn.predict(x_test_scaled)

ridge = Ridge(alpha=1.0)
ridge.fit(x_train_scaled, y_train_scaled)
joblib.dump(ridge, "models/ridge.pkl")
ridge_pred = ridge.predict(x_test_scaled)

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
gbr.fit(x_train_scaled, y_train_scaled.ravel())
joblib.dump(gbr, "models/gbr.pkl")
gbr_pred = gbr.predict(x_test_scaled)

# ========== Quantile Regression ==========
quantiles = [0.1, 0.5, 0.9]
quantile_models = {}
for q in quantiles:
    gbr_q = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=300, max_depth=4, random_state=42)
    gbr_q.fit(x_train_scaled, y_train.ravel())
    quantile_models[q] = gbr_q
    joblib.dump(gbr_q, f"models_quantile/quantile_model_{int(q*100)}.pkl")

q10 = quantile_models[0.1].predict(x_test_scaled)
q50 = quantile_models[0.5].predict(x_test_scaled)
q90 = quantile_models[0.9].predict(x_test_scaled)

q10_inv = y_scaler.inverse_transform(q10.reshape(-1, 1))
q50_inv = y_scaler.inverse_transform(q50.reshape(-1, 1))
q90_inv = y_scaler.inverse_transform(q90.reshape(-1, 1))

band_width = (q90 - q10).reshape(-1, 1)  # Scaled
median_pred = q50.reshape(-1, 1)         # Scaled

# ========== Build Meta-Model Features ==========
# Stack: [knn, ridge, gbr, q50, (q90 - q10)]
meta_features = np.hstack([
    knn_pred.reshape(-1, 1),
    ridge_pred.reshape(-1, 1),
    gbr_pred.reshape(-1, 1),
    median_pred,
    band_width
])

# Meta-Learner
meta_model = LinearRegression()
meta_model.fit(meta_features, y_test_scaled.ravel())
meta_pred_scaled = meta_model.predict(meta_features)
meta_pred = y_scaler.inverse_transform(meta_pred_scaled.reshape(-1, 1))
y_test_orig = y_scaler.inverse_transform(y_test_scaled)

# Save meta-model
joblib.dump(meta_model, "models/meta_learner_stacked.pkl")

# ========== Evaluation ==========
rmse = np.sqrt(mean_squared_error(y_test_orig, meta_pred))
r2 = r2_score(y_test_orig, meta_pred)
print(f"Stacked Ensemble RMSE: {rmse:.2f}")
print(f"Stacked Ensemble R²: {r2:.3f}")

# ========== Save Predictions ==========
np.savetxt("predictions/stacked_ensemble_predictions.csv", meta_pred, delimiter=",")
np.savetxt("predictions/y_test_actual.csv", y_test_orig, delimiter=",")
np.savetxt("predictions/quantile_q10.csv", q10_inv, delimiter=",")
np.savetxt("predictions/quantile_q50.csv", q50_inv, delimiter=",")
np.savetxt("predictions/quantile_q90.csv", q90_inv, delimiter=",")

# ========== Plot: Ensemble vs Actual ==========
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, meta_pred, alpha=0.6, label="Stacked Ensemble Prediction")
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', label="Ideal")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Stacked Ensemble: Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions/stacked_ensemble_predictions.pdf")
plt.close()

# ========== Plot: Quantile Band ==========
plt.figure(figsize=(10, 6))
plt.plot(y_test_orig, label='Actual', alpha=0.7)
plt.plot(q50_inv, label='Predicted Median (Q50)', color='blue')
plt.fill_between(np.arange(len(q10_inv)), q10_inv.ravel(), q90_inv.ravel(), color='blue', alpha=0.2, label='Q10 - Q90 range')
plt.title("Quantile Regression Bands with Stacked Ensemble")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions/quantile_prediction_bands_stacked.pdf")
plt.close()