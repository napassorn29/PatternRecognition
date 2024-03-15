import numpy as np

def sigmoid(x):
# Description: Calculate sigmoid([x]).
    return 1/(1 + np.exp(-x))

def logistic_fit(X: np.ndarray, Y: np.ndarray, r: float, threshold: float):
# Description: Fit the model with given input [X] and output [y] by adjusting the weight of logistic function with learning rate [r].
#              The model will update iteratively until the change of MSE is smaller than [threshold].

    # -- |Local variables initialization| --
    _X = np.insert(X, 0, 1, axis = 1)       # local X will have an extra column of 1 for theta_0 which are bias
    theta = np.zeros((_X[0].size,1))        # parameters
    cross_entropy = threshold + 1           # Binary-Cross Entropy
    cross_entropy_last = 0                  # Last iteration Binary-Cross Entropy 
    cross_entropy_history = []              # Binary-Cross Entropy update history
    eps = 1e-15                             # Very small number to prevent any mathematically undefine term with cause by zero

    while np.abs(cross_entropy - cross_entropy_last) > threshold:
        # -- |Gradient calculation (Batch update)| --
        gradiant = (2/X.shape[0])*(_X.T @ (Y - sigmoid(_X @ theta)))
        
        # -- |Parameters update| --
        theta = theta + r*gradiant

        # -- |Update last cross entropy| --
        cross_entropy_last = cross_entropy 

        # -- |Cross entropy calculation| --
        h = sigmoid(_X @ theta)
        cross_entropy = - (np.sum(Y.T @ np.log(h + eps)) + np.sum((1 - Y).T @ np.log(1 - h + eps)))/_X.shape[0]
        cross_entropy_history.append(cross_entropy)

    return theta, cross_entropy_history

def linear_fit(X: np.ndarray, Y: np.ndarray, r: float, threshold: float):
# Description: Fit the model with given input [X] and output [y] by adjusting the weight of linear function with learning rate [r].
#              The model will update iteratively until the change of MSE is smaller than [threshold].

    # -- |Local variables initialization| --
    _X = np.insert(X, 0, 1, axis = 1)       # local X will have an extra column of 1 for theta_0 which are bias
    theta = np.zeros((X[0].size + 1,1))     # parameters
    MSE = threshold + 1                     # Mean Square Error
    MSE_last = 0                            # Last Mean Square Error
    MSE_history = []                        # Mean Square Error update history

    while np.abs(MSE - MSE_last) > threshold:
        # -- |Gradient calculation (Batch update)| --
        gradiant = (2/X.shape[0])*(_X.T @ (Y - (_X @ theta)))
        
        # -- |Parameters update| --
        theta = theta + r*gradiant

        # -- Update Last MSE| --
        MSE_last = MSE

        # -- |MSE calculation| --
        MSE = np.sum((Y - (_X @ theta))**2)/_X.shape[0]
        MSE_history.append(MSE)

    return theta, MSE_history

def matrix_linear_fit(X: np.ndarray, Y: np.ndarray):
# Description: Solve the parameter use to map the input [X] to output [Y] using matrix inversion method.
    
    # -- |Local variable initialization| --
    _X = np.insert(X, 0, 1, axis = 1)       # local X will have an extra column of 1 for theta_0 which are bias
    
    # -- |Matrix inversion| --
    theta = np.linalg.inv(_X.T @ _X) @ _X.T @ Y

    # -- |MSE calculation| --
    MSE = np.sum((Y - (_X @ theta))**2)/_X.shape[0]

    return theta, MSE

def logistic_transform(X: np.ndarray, theta: np.ndarray, threshold: float):
# Description: Compute the logistic function for given input [X] and parameters [theta].
#              The output will be determined as 1 (True) if logistic function output is greater than [threshold].
    out = []
    for i in range(X.shape[0]):
        h = sigmoid(((np.transpose(theta[1:]) @ np.transpose([X[i]]))[0][0] + theta[0][0]))
        if h >= threshold:
            out.append([1])
        else:
            out.append([0])
    return np.array(out)

def linear_transform(X: np.ndarray, theta: np.ndarray):
# Description: Compute the linear function for given input [X] and parameters [theta].
    out = []
    for i in range(X.shape[0]):
        out.append([((np.transpose(theta[1:]) @ np.transpose([X[i]]))[0][0] + theta[0][0])])
    return np.array(out)

def logistic_display(theta: np.ndarray):
# Description: Retuirn string display of logistic model with given parameters [theta].
    display = 'y = sigmoid({0:2f}'.format(theta[0][0])
    for t, i in zip(theta.T[0][1:], range(1, len(theta.T[0][1:]) + 1)):
        if t >= 0:
            display += ' + {0:2f}x_{1:1d}'.format(t, i)
        else:
            display += ' - {0:2f}x_{1:1d}'.format(-t, i)
    display += ')'
    return display

def linear_display(theta: np.ndarray):
# Description: Retuirn string display of linear model with given parameters [theta].
    display = 'y = {0:2f}'.format(theta[0][0])
    for t, i in zip(theta.T[0][1:], range(1, len(theta.T[0][1:]) + 1)):
        if t >= 0:
            display += ' + {0:2f}x_{1:1d}'.format(t, i)
        else:
            display += ' - {0:2f}x_{1:1d}'.format(-t, i)
    return display