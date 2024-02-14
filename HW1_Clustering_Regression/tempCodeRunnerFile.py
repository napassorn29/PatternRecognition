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