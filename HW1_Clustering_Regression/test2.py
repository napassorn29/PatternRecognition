import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.0001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def compute_cost(self, X, y):
        m = len(y)
        predictions = np.dot(X, self.weights) + self.bias
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def gradient_descent(self, X, y):
        m = len(y)
        predictions = np.dot(X, self.weights) + self.bias
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1) * 0.01  # Initialize weights with small random values
        self.bias = 0
        
        for _ in range(self.num_iterations):
            dw, db = self.gradient_descent(X, y)
            self.weights -= self.learning_rate * dw.reshape(-1, 1)  # Reshape dw for proper broadcasting
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Assuming x_train and y_train are your training data and labels
x_train = np.array([[1., 0., 38., 1.],
                    [1., 0., 35., 0.],
                    [1., 1., 54., 0.],
                    [3., 0., 4., 0.],
                    # Add more rows...
                    [1., 1., 24., 1.]])

y_train = np.array([[1], [1], [0], [1], # Add more labels...
                    [0]])

# Instantiate and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Compute MSE for training set
train_predictions = model.predict(x_train)
train_loss = model.compute_cost(x_train, y_train)

# Print weights and loss
print("Weights:", model.weights)
print("Bias:", model.bias)
print("Training Set Loss (MSE):", train_loss)

mse_train = np.mean((train_predictions - y_train) ** 2)
print("Mean Squared Error on training data:", mse_train)

predictions = model.predict(x_train)

plt.scatter(x_train[:, 2], y_train, color='blue', label='Actual')
plt.plot(x_train[:, 2], predictions, color='red', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Linear Regression')
plt.legend()
plt.show()