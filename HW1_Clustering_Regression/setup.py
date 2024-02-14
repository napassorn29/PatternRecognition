import numpy as np
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot


class Clustering():
    def __init__(self, k:int, initial_centroids:np.array, points:np.array):
        if k == len(initial_centroids):
            self.k = k
            self.initial_centroids = initial_centroids
            self.points = points
        
        else:
            raise ValueError('Number of initial centroids must match k')
            
    # Function to calculate Euclidean distance between two points
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Function to perform K-means clustering
    def k_means(self, iterations):
        centroid_updated = self.initial_centroids
        if iterations == 0:
            clusters = [[] for j in range(len(centroid_updated))]
            for point in self.points:
                distance = []
                for centroid in centroid_updated:
                    dist = self.euclidean_distance(point, centroid)
                    distance.append(dist)
                closest_centroid_index = np.argmin(distance)
                clusters[closest_centroid_index].append(point)
            return clusters, centroid_updated
        
        else :
            for i in range(iterations):
                # Assignment step
                clusters = [[] for j in range(len(centroid_updated))]
                for point in self.points:
                    distance = []
                    for centroid in centroid_updated:
                        dist = self.euclidean_distance(point, centroid)
                        distance.append(dist)
                    closest_centroid_index = np.argmin(distance)
                    clusters[closest_centroid_index].append(point)
                
                # Update step
                for i in range(len(self.initial_centroids)):
                    if clusters[i]:
                        centroid_mean = np.mean(clusters[i], axis=0)
                        centroid_updated[i] = centroid_mean
            
            return clusters, centroid_updated
        
    def calculate_wcss(self, data, k):
        indices = np.random.choice(range(len(data)), size=k, replace=False)
        centroid_init = data[indices]
        
        clusters = [[] for _ in range(len(centroid_init))]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroid_init]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(point)
            
        wcss = 0
        for i in range(len(centroid_init)):
            centroid = centroid_init[i]
            cluster = clusters[i]
            for point in cluster:
                wcss += self.euclidean_distance(point, centroid) ** 2
            
        return wcss

    def elbow_method(self, k_values:int):
        wcss_values = []
        for k in range(k_values):
            wcss = self.calculate_wcss(self.points, k+1)
            if k == 0:
                wcss_first = wcss
            wcss = wcss/wcss_first * 100
            if wcss > 10:
                k_select = k+2
            wcss_values.append(wcss)
        return wcss_values,k_select

class Regression:
    def __init__(self,mode:str = 'logistic', learning_rate=0.01, num_iterations=1000):
        self.mode = mode
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # Store the cost history during training
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y, y_predicted):
        # Binary Cross-Entropy Loss
        epsilon = 1e-15  # Small value to prevent log(0)
        cost = - np.mean(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon))
        return cost

    def gradient_descent(self, X, y):
        num_samples = len(y)
        # Linear combination
        linear_model = np.dot(X, self.weights) + self.bias
        # Sigmoid function
        if self.mode == 'logistic':
            y_predicted = self.sigmoid(linear_model)
        elif self.mode == 'linear':
            y_predicted = linear_model

        # Compute gradients
        dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / num_samples) * np.sum(y_predicted - y)
        return dw, db, y_predicted
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, 1)) 
        self.bias = 0                
        
        # Gradient Descent
        for _ in range(self.num_iterations):
            dw, db, y_predicted = self.gradient_descent(X, y)

            # Update parameters
            self.weights -= self.learning_rate * dw.reshape(-1, 1)
            self.bias -= self.learning_rate * db

            if self.mode == 'logistic':
                # Compute and store the cost
                self.cost = self.compute_cost(y, y_predicted)
                self.cost_history.append(self.cost)
            elif self.mode == 'linear':
                # Compute and store the cost
                y_pre = self.sigmoid(y_predicted)
                self.cost = self.compute_cost(y,y_pre)
                self.cost_history.append(self.cost)
                # self.cost = 'no cost for linear regression'
                # self.cost_history.append(self.cost)
                
    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        if self.mode == 'logistic':
            predict_probabity = self.sigmoid(linear_model)
        if self.mode == 'linear':
            predict_probabity = linear_model
        return predict_probabity
    
    def predict(self, X, threshold=0.5):
        # Predict probabilities
        list_prob = []
        list_predict = []
        probabilities = self.predict_prob(X)
        list_prob.append(probabilities)
        # Threshold probabilities to obtain binary predictions
        for prob in probabilities:
            if prob >= threshold:
                predictions = 1
                list_predict.append(predictions)
            else:
                predictions = 0
                list_predict.append(predictions)
        return predictions,list_predict,list_prob
    
    def RMS_error(self,true,pred):
        mse_train = np.mean((pred - true) ** 2)
        return mse_train
    
    def Matrix_inversion(self, X, y, threshold=0.5):
        num_samples, num_features = X.shape
        X = X.reshape((len(X), num_features))
        # QR decomposition
        Q, R = qr(X)
        b = inv(R).dot(Q.T).dot(y)
        yhat = X.dot(b)
        for prob in yhat:
            if prob >= threshold:
                predictions = 1
                list_predict.append(predictions)
            else:
                predictions = 0
                list_predict.append(predictions)
        return predictions,list_predict,np.array(yhat)
        # X_transpose = X.T
        # X_transpose_X = np.dot(X_transpose, X)
        # X_transpose_X_inv = np.linalg.inv(X_transpose_X)    
        # return np.dot(np.dot(X_transpose_X_inv, X_transpose), y)

class Accuracy():
    def __init__(self, true, pred):
        self.true = true
        self.pred = pred
    
    def accuracy(self):
        correct = sum(1 for t, p in zip(self.true, self.pred) if t == p)
        total = len(self.true)
        accuracy = correct / total
        return accuracy

    def precision(self):
        true_positive = sum(1 for t, p in zip(self.true, self.pred) if t == 1 and p == 1)
        predicted_positive = sum(1 for p in self.pred if p == 1)
        if predicted_positive == 0:
            return 0
        precision = true_positive / predicted_positive
        return precision

    def recall(self):
        true_positive = sum(1 for t, p in zip(self.true, self.pred) if t == 1 and p == 1)
        actual_positive = sum(1 for t in self.true if t == 1)
        if actual_positive == 0:
            return 0
        recall = true_positive / actual_positive
        return recall

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        # Compute F1 score
        if prec + rec == 0:
            return 0
        f1_score = 2 * (prec * rec) / (prec + rec)
        return f1_score
    

# Example usage
x_train = np.array([[1., 0., 38., 1.],
                    [1., 0., 35., 0.],
                    [1., 1., 54., 0.],
                    [3., 0., 4., 0.],
                    # Add more rows...
                    [1., 1., 24., 1.]])

y_train = np.array([[1], [1], [0], [1], # Add more labels...
                    [0]])

# Instantiate and train the model
model = Regression('linear',0.0012,50000)
model.fit(x_train, y_train)

# Plot the cost function
import matplotlib.pyplot as plt

plt.plot(range(1, model.num_iterations + 1), model.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

print(model.cost)
train_predictions,list_predict,list_prob = model.predict(x_train)
print(list_prob)
print("RMSE",model.RMS_error(y_train, list_prob))

Accuracy_model = Accuracy(y_train, list_predict)

print("Accuracy :", Accuracy_model.accuracy())
print("Precision :", Accuracy_model.precision())
print("Recall :", Accuracy_model.recall())
print("f1_score :", Accuracy_model.f1_score())


model1 = Regression()
predictions,list_predict,yhat = model1.Matrix_inversion(x_train, y_train)
print("RMSE",model.RMS_error(y_train, yhat))
# train_predictions,list_predict,list_prob = model.predict(yhat)
print(yhat)
# Plotting
plt.scatter(x_train[:, 2], y_train, label='Actual')
plt.plot(x_train[:, 2], yhat, color='red', label='Predicted')
plt.xlabel('Feature 3')
plt.ylabel('Target')
plt.legend()
plt.show()
# # Make sure x_train and y_train have the same length
# if len(x_train) == len(y_train):
#     plt.scatter(x_train, y_train)
#     plt.plot(x_train, yhat, color='red')
#     plt.show()
# else:
#     print("Error: x_train and y_train must have the same length.")

# pyplot.scatter(x_train, y_train)
# pyplot.plot(x_train, yhat, color='red')
# pyplot.show()


