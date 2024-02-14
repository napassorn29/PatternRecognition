import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # Store the loss history during training
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, 1))  # Initialize weights as a column vector
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (linear_model - y))
            db = (1 / num_samples) * np.sum(linear_model - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and store the loss
            loss = np.mean((linear_model - y) ** 2)
            self.loss_history.append(loss)
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Visualize the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Randomly Generated Data')
plt.show()
# Load the Titanic dataset and preprocess it
# Assuming you have X_train, y_train, X_test, y_test already prepared

# Instantiate and train the linear regression model
model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, predictions, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()

# Print weights and bias
print("Weights:", model.weights)
print("Bias:", model.bias)

# Plot the loss over iterations
plt.plot(range(1, model.num_iterations + 1), model.loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()