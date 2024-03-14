import numpy as np
from scipy.stats import norm


class NaiveBayes:
    def __init__(self, mode = 'histogram'):
        self.prior = None
        # mode for histogram or probability density function
        self.mode = mode
        # variables for likelihood
        self.likelihood_categorical = None
        self.likelihood_numerical = None
        # variables for probability density function
        self.mean = None
        self.std = None

    def fit(self, X_train, y_train, numeric_features, cat_features, bin:list = None):
        self.cat_features = cat_features
        self.numeric_features = numeric_features
        self.prior = y_train.value_counts(normalize=True)
        
        if self.mode == 'histogram':
            # variables for likelihood
            self.likelihood_categorical = {}
            self.likelihood_numerical = {}
    
            # Calculate likelihood for each feature
            for feature in X_train.columns:
                # calculate likelihood for numerical features
                if feature in numeric_features:
                    self.likelihood_numerical[feature] = {}
                    index_bin = numeric_features.index(feature)
                    # calculate likelihood for each category in the feature
                    for category in y_train.unique():
                        # Drop NaN values
                        values_train = X_train[feature][y_train == category].dropna()
                        # histogram for likelihood
                        values, bins_all = np.histogram(X_train[feature].dropna(), bins=bin[index_bin], density=True)
                        values, bins = np.histogram(values_train, bins=bins_all, density=True)
                        prob_values = values/np.sum(values)
                        # keep the likelihood value for each category
                        self.likelihood_numerical[feature][category] = (prob_values, bins)
                        
                # calculate likelihood for categorical features
                if feature in cat_features:
                    self.likelihood_categorical[feature] = {}
                    # calculate likelihood for each category in the feature
                    for category in y_train.unique():
                        self.likelihood_categorical[feature][category] = (
                            X_train[feature][y_train == category].value_counts(normalize=True)
                        )
                        
        elif self.mode == 'pdf':
            # variables for probability density function
            self.mean = {}
            self.std = {}
            self.likelihood_categorical = {}
            # Calculate likelihood for each feature
            for feature in X_train.columns:
                # calculate likelihood for numerical features
                if feature in numeric_features:
                    self.mean[feature] = {}
                    self.std[feature] = {}
                    # calculate likelihood for each category in the feature
                    for category in y_train.unique():
                        # Drop NaN values
                        values_train = X_train[feature][y_train == category].dropna()
                        # probability density function for likelihood
                        self.mean[feature][category] = values_train.mean()
                        self.std[feature][category] = values_train.std()
                
                # calculate likelihood for categorical features
                if feature in cat_features:
                    self.likelihood_categorical[feature] = {}
                    # calculate likelihood for each category in the feature
                    for category in y_train.unique():
                        self.likelihood_categorical[feature][category] = (
                            X_train[feature][y_train == category].value_counts(normalize=True)
                        )
            
    # Predict the class of each sample
    def predict(self, X_test, normal_tresholds=1, log_tresholds=0):
        
        p_leave = self.prior[1]
        p_stay = self.prior[0]
        if self.mode == 'histogram':
            predictions = []
            predictions_log = []
            h_x_list = []
            log_h_x_list = []
            
            for _, row in X_test.iterrows():
                posterior_list = []
                posterior_list_log = []
                
                for feature, value in row.items():
                # Calculate p(xi | leave), p(xi | stay) for categorical features
                    if feature in self.cat_features:
                        if value in self.likelihood_categorical[feature][1]:
                            posterior_leave = self.likelihood_categorical[feature][1][value]
                        else:
                            posterior_leave = 0.00000000000000000001  # Add smoothing if value not in the feature in training data
                        if value in self.likelihood_categorical[feature][0]:
                            posterior_stay = self.likelihood_categorical[feature][0][value]
                        else:
                            posterior_stay = 0.00000000000000000001  # Add smoothing if value not in the feature in training data
                
                    else:
                    # Skip NaN values
                        if np.isnan(row[feature]):
                            posterior_leave = 0.0000001 # Add smoothing if data is Nan
                            posterior_stay = 0.0000001  # Add smoothing if data is Nan
                            continue
                        # classify the mode for histogram or probability density function
                        else:
                            # calduclate p(xi | leave), p(xi | stay) for each category in each feature
                            # + 0.0000000001 to avoid zero probability
                            posterior_leave = np.interp(row[feature], self.likelihood_numerical[feature][1][1][:-1],
                                                        self.likelihood_numerical[feature][1][0]) + 0.0000000001
                            posterior_stay = np.interp(row[feature], self.likelihood_numerical[feature][0][1][:-1],
                                                    self.likelihood_numerical[feature][0][0]) + 0.0000000001
                            
                    posterior = posterior_leave/posterior_stay
                    posterior_list.append(posterior)
                    # calculate the log (logp(xi | leave) - logp(xi | stay))
                    posterior_log = np.log(posterior_leave) - np.log(posterior_stay)
                    posterior_list_log.append(posterior_log)
                    
                mul_posterior = 1
                for num in posterior_list:
                    mul_posterior *= num
                # calculate h(x) = (p_leave / p_stay) * p(xi | leave) / p(xi | stay) for each sample
                h_x = (p_leave / p_stay) * mul_posterior
                h_x_list.append(h_x)
                # classify the tresholds
                if h_x > normal_tresholds: predictions.append([1])
                else: predictions.append([0])    
                    
                # calculate the log (logp(xi | leave) - logp(xi | stay)) by sum all features
                sum_posterior = sum(posterior_list_log)
                # calculate h(x) = (p_leave - p_stay) + (logp(xi | leave) - logp(xi | stay)) for each sample
                log_h_x = (np.log(p_leave) - np.log(p_stay)) + sum_posterior
                log_h_x_list.append(log_h_x)
                # classify the tresholds
                if log_h_x > log_tresholds: predictions_log.append([1])
                else: predictions_log.append([0])
            
            return np.array(predictions, dtype =int), np.array(predictions_log, dtype =int), h_x_list, log_h_x_list   
                
        elif self.mode == 'pdf':
            predictions_pdf = []
            log_h_x_list = []
            
            for _, row in X_test.iterrows():
                posterior_list = []
                for feature, value in row.items():
                # Calculate p(xi | leave), p(xi | stay) for categorical features
                    if feature in self.cat_features:
                        if value in self.likelihood_categorical[feature][1]:
                            posterior_leave = self.likelihood_categorical[feature][1][value]
                        else:
                            posterior_leave = 0.00000000000000000001  # Add smoothing if value not in the feature in training data
                        if value in self.likelihood_categorical[feature][0]:
                            posterior_stay = self.likelihood_categorical[feature][0][value]
                        else:
                            posterior_stay = 0.00000000000000000001  # Add smoothing if value not in the feature in training data
                            
                    else:
                    # Skip NaN values
                        if np.isnan(row[feature]):
                            posterior_leave = 0.0000001 # Add smoothing if data is Nan
                            posterior_stay = 0.0000001  # Add smoothing if data is Nan
                            continue
                        # classify the mode for histogram or probability density function
                        else:
                            posterior_leave = norm.pdf(value, loc=self.mean[feature][1], scale=self.std[feature][1])
                            posterior_stay = norm.pdf(value, loc=self.mean[feature][0], scale=self.std[feature][0])
                            
                    posterior_log = np.log(posterior_leave) - np.log(posterior_stay)
                    posterior_list.append(posterior_log)
                    
                # calculate the log (logp(xi | leave) - logp(xi | stay)) by sum all features
                sum_posterior = sum(posterior_list)
                # calculate h(x) = (p_leave - p_stay) + logp(xi | leave) - logp(xi | stay) for each sample
                log_h_x = (np.log(p_leave) - np.log(p_stay)) + sum_posterior
                log_h_x_list.append(log_h_x)
                # classify the tresholds            
                if log_h_x > log_tresholds: predictions_pdf.append([1])
                else: predictions_pdf.append([0])
            
            return np.array(predictions_pdf, dtype =int),log_h_x_list
        