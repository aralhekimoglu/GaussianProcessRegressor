import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class GPRegressor:
    def __init__(self, X, y, noise=0.02):
        self.N = len(X)
        self.X = X
        self.y = y.reshape(self.N,1)
        self.noise = noise
    
    def predict(self, x_pred):
        """
            Compute posterior mean and covariance to return a prediction.
        Args:
            x_pred(np.array): x value to predict
        Returns: 
            predictionMean(np.array): Return the posterior mean for given data and x_pred
            predictionCovariance(np.array): Return the posterior mean for given data and x_pred
        """
        K_11 = seKernel(self.X)+ self.noise * np.eye(self.N)
        K_12 = np.array([squaredExp(self.X[i], x_pred) for i in range (self.N)])
        K_22=squaredExp(x_pred, x_pred)+self.noise
        K_11_inv=np.linalg.inv(K_11)   
        
        predictionMean = K_12.T.dot(K_11_inv).dot(self.y)
        predictionCovariance = K_22-K_12.T.dot(K_11_inv).dot(K_12)
        
        return predictionMean, predictionCovariance
    
    def plotTest(self):
        """
        Plot the resulting gaussian process. For a given x, mean+-3*covariance.
        Args:
            None.
        Returns: 
            None.
        """
        x = np.linspace(-2, 2, 400)
        y,covariance = np.vectorize(self.predict)(x)
        plt.plot(x, y, c="b")
        plt.plot(x, y + 3*np.sqrt(covariance), "r:")
        plt.plot(x, y - 3*np.sqrt(covariance), "r:")
    
def squaredExp(x1, x2, tho=1.0):
    """
        Squared exponential function.
        Args:
            x1(np.array)
            x2(np.array)
            tho(float): Hyperparameter for the function
        Returns: 
            (np.array): value of function for given inputs.
    """
    return np.exp(np.power(x1-x2, 2) / (-2.*tho**2) )

def seKernel(X):
    """
        Construct a squared exponential kernel for given X.
        Args:
            X(np.array): X to use in kernel construction
        Returns: 
            Kernel(np.array): RBF Kernel for given X.
    """
    N=X.shape[0]
    Kernel = np.ones((N,N))
    for i in range(N):
        for j in range(i+1, N):
            k_ij = squaredExp(X[i],X[j])
            Kernel[i][j] = k_ij
            Kernel[j][i] = k_ij
    return Kernel   

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
plt.scatter(x, y, c="black")

regressor = GPRegressor(x, y)
regressor.plotTest()

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))[0])[0])