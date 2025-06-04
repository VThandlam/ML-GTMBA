import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load datasets
#dataset1 = loadtxt('sw_training.csv', delimiter=',', skiprows=1)
dataset2 = loadtxt('sw_test.csv', delimiter=',', skiprows=1)

# Split into input (X) and output (Y) variables
X = dataset2[:, 0:10]
Y = dataset2[:, 10]

# Scale input data
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()
X_scaled = input_scaler.fit_transform(X)
#X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  # Reshape X for the model
Y_scaled = output_scaler.fit_transform(Y.reshape(-1, 1))  # Reshape not necessary for Y

# Load pre-trained model
model = load_model('sw_model.keras')

# Make predictions
predictions = model.predict(X_scaled)
predictions_inverse_transformed = output_scaler.inverse_transform(predictions)

# Concatenate original inputs and predictions
output_with_inputs = np.hstack((X, predictions_inverse_transformed))

# Define format string for entire array
fmt = ['%d', '%d'] + ['%1.3f'] * (output_with_inputs.shape[1] - 2)


# Save results to CSV
np.savetxt("sw_test_results.csv", output_with_inputs, delimiter=",", fmt=fmt)

print("Output file saved successfully.")
print(X.shape, output_with_inputs.shape)
