# KNN for regression with enhanced features
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
dataset = loadtxt('../global_sw_train.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:7]
Y = dataset[:, 7].reshape(-1, 1)

# Shuffle and split
X, Y = shuffle(X, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create scalers
input_scaler = RobustScaler()
output_scaler = RobustScaler()

# Scale inputs
x_train_scaled = input_scaler.fit_transform(x_train)
x_test_scaled = input_scaler.transform(x_test)

# Scale outputs
y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

# GridSearchCV for hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan and Euclidean distances
}

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(x_train_scaled, y_train_scaled.ravel())

best_knn = grid_search.best_estimator_
print(f"Best KNN parameters: {grid_search.best_params_}")

# Cross-validated performance on train set
cv_scores = cross_val_score(best_knn, x_train_scaled, y_train_scaled.ravel(), cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validated RMSE (mean ± std): {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

# Predictions
train_pred_scaled = best_knn.predict(x_train_scaled)
test_pred_scaled = best_knn.predict(x_test_scaled)

# Inverse transform
train_pred = output_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1))
test_pred = output_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1))
y_test_orig = output_scaler.inverse_transform(y_test_scaled)

# Metrics
mae = mean_absolute_error(y_test_orig, test_pred)
mse = mean_squared_error(y_test_orig, test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, test_pred)

print(f"\nEvaluation Metrics on Test Set:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# Save metrics
with open("knn_evaluation_metrics.txt", "w") as f:
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write(f"Cross-validated RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.3f}\n")

# Plot: Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, test_pred, c='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()],
         'r--', label='Ideal Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('KNN Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_prediction_improved.pdf")
plt.show()

# Plot: Residuals
residuals = y_test_orig - test_pred
plt.figure(figsize=(10, 4))
plt.scatter(test_pred, residuals, alpha=0.6, color='darkgreen')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_residuals_plot.pdf")
plt.show()

# Save predictions
np.savetxt("knn_improved_x_test_predictions.csv", test_pred, delimiter=",")
np.savetxt("knn_improved_y_test.csv", y_test_orig, delimiter=",")

# Save model and scalers
joblib.dump(best_knn, "knn_model_improved.pkl")
joblib.dump(input_scaler, "input_scaler_improved.pkl")
joblib.dump(output_scaler, "output_scaler_improved.pkl")
