# KNN for regression with scikit-learn
import numpy as np
from numpy import loadtxt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load dataset
dataset = loadtxt('../global_lw_train.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:7]
Y = dataset[:, 7].reshape(-1, 1)

# Shuffle and split data
X, Y = shuffle(X, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scaling input and output
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()

x_train_scaled = input_scaler.fit_transform(x_train)
x_test_scaled = input_scaler.transform(x_test)

y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

# Define KNN model
knn = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')
knn.fit(x_train_scaled, y_train_scaled.ravel())

# Evaluate
train_pred_scaled = knn.predict(x_train_scaled)
test_pred_scaled = knn.predict(x_test_scaled)

train_mse = mean_squared_error(y_train_scaled, train_pred_scaled)
test_mse = mean_squared_error(y_test_scaled, test_pred_scaled)

print(f"Train RMSE: {np.sqrt(train_mse):.2f}")
print(f"Test RMSE: {np.sqrt(test_mse):.2f}")

# Inverse transform predictions and targets
train_pred = output_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1))
test_pred = output_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1))
y_test_orig = output_scaler.inverse_transform(y_test_scaled)

# Save predictions
np.savetxt("knn_jan_x_test_predictions.csv", test_pred, delimiter=",")
np.savetxt("knn_jan_y_test.csv", y_test_orig, delimiter=",")

# Save the model and scalers
joblib.dump(knn, "knn_model.pkl")
joblib.dump(input_scaler, "input_scaler.pkl")
joblib.dump(output_scaler, "output_scaler.pkl")

# Plot predictions vs truth
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, test_pred, c='blue', label='Predicted vs Actual')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', label='Ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('KNN Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_prediction_plot.pdf")
plt.show()