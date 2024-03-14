import numpy as np

class normal_dist:
# Description : Class of normaldistribution function which contains [mean] and [variance] as parameters.
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def get_p(self, x):
        #  Description : Return the probability of X = [x] given the defined distribution.
        return (np.exp(-(x-self.mean)**2/(2*self.variance)))/(np.sqrt(2*np.pi*self.variance))
    
def decision_boundary(N1: normal_dist, N2: normal_dist, prior_ratio: float = 1):
# Description : Solve for x which makes the ratio between the normal distribution [N1] and normal distribution [N2] equal to [prior_ratio].
    output = np.nan
    if N1.variance == N2.variance:
        output = (N2.mean**2 - N1.mean**2 - 2*N1.variance*np.log(prior_ratio))/(2*(N2.mean - N1.mean))
    else:
        inRoot = ((N1.mean**2+N2.mean**2) + (N2.variance-N1.variance)*np.log(N2.variance/(N1.variance*prior_ratio**2)) - 2*N1.mean*N2.mean)*N1.variance*N2.variance
        if N1.mean > N2.mean and inRoot >= 0:
            output = (N1.mean*N2.variance - N2.mean*N1.variance - inRoot**0.5)/(N2.variance - N1.variance)
        elif N1.mean < N2.mean and inRoot >= 0:
            output = (N1.mean*N2.variance - N2.mean*N1.variance + inRoot**0.5)/(N2.variance - N1.variance)
    return output

def LRT(x: float, N1: normal_dist, N2: normal_dist, prior_ratio: float = 1):
# Description : Perform a likelihood ratio test. The function classify which hypothesis is based on the likelihood of class 1 [N1] and class 2 [N2], 
#               and [prior_ratio] of class 2 by class 1 given the evidence [x].
#               An output equal to 1 indicates that x is more likely to belong to class 1 than class 2, and vice versa.
    output = np.nan
    if N1.get_p(x)/N2.get_p(x) > prior_ratio:
        output = 1
    elif N1.get_p(x)/N2.get_p(x) < prior_ratio:
        output = 2
    else:
        output = np.random.randint(0,2)
    return output