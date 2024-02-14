import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.inv = None
        
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.inv = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        # Compute pseudo-inverse
        X_pseudo_inv = np.linalg.pinv(X_b)
        # Calculate weights
        self.weights = np.dot(X_pseudo_inv, y)
        
    def predict(self, X):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Predict using the calculated weights
        return np.dot(X_b, self.weights)

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([[0], [0], [1], [1]])

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on training data
predictions_train = model.predict(X_train)
print("Predictions on training data:")
print(predictions_train)

print("data:")
print(model.inv)

print("y_pred", predictions_train)
print("y_train",y_train)
# Evaluate the model using mean squared error
mse_train = np.mean((predictions_train - y_train) ** 2)
print("Mean Squared Error on training data:", mse_train)