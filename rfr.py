#Random Forest Regressor with RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import joblib

# Load dataset
dataset = np.loadtxt('../global_sw_train.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:7]
Y = dataset[:, 7].reshape(-1, 1)

# Shuffle and split
X, Y = shuffle(X, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale inputs and outputs
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()

x_train_scaled = input_scaler.fit_transform(x_train)
x_test_scaled = input_scaler.transform(x_test)

y_train_scaled = output_scaler.fit_transform(y_train).ravel()
y_test_scaled = output_scaler.transform(y_test).ravel()

# Define the model and parameter grid
rf = RandomForestRegressor(random_state=42, oob_score=True)

param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 40, 60, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5]
}

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

# Fit model
random_search.fit(x_train_scaled, y_train_scaled)

# Save best model
joblib.dump(random_search.best_estimator_, 'rf_randomized_best_model.pkl')

# Predictions
predictions_scaled = random_search.predict(x_test_scaled)
predictions = output_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
y_test_orig = output_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

# Save predictions
np.savetxt("jan_rf_predictions.csv", predictions, delimiter=",")
np.savetxt("jan_rf_y_test.csv", y_test_orig, delimiter=",")

# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_orig, predictions)
mse = mean_squared_error(y_test_orig, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, predictions)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# Plot predictions vs true
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, predictions, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Random Forest Predictions vs True Values")
plt.savefig("jan_rf_prediction_vs_true.pdf")
plt.show()

# Plot feature importances
importances = random_search.best_estimator_.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.savefig("jan_rf_feature_importance.pdf")
plt.show()

