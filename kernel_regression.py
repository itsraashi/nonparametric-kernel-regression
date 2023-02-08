import numpy as np
import numpy.linalg

def load_data():
    """ Load training dataset

        Returns tuple of length 4: (X_train, y_train, X_val, y_val)
        where X_train is an N_train-x-M ndarray and y_train is an N_train-x-1 ndarray and
        where X_val is an N_val-x-M ndarray and y_val is an N_val-x-1 ndarray.
    """
    X = np.load('data/regression_train_input.npy')
    y = np.load('data/regression_train_output.npy')

    N = len(y)
    N_val = 10
    N_train = N - N_val    

    X_train = X[:N_train]
    y_train = y[:N_train]
    X_val = X[N_train:]
    y_val = y[N_train:]

    return (X_train, y_train, X_val, y_val)


def kernel_boxcar(x, z, h):
    """ Return the result of applying the boxcar kernel on the two input vectors.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        
        Returns: float value after appying kernel to x and z
    """
    normed = numpy.linalg.norm(x-z)
    if normed <= (h / 2):
        return 1.0
    return 0.0

def kernel_rbf(x, z, h):
    """ Return the result of applying the radial basis function kernel 
        on the two input vectors, given the hyperparameter h.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        gamma: float value of hyperparameter
        
        Returns: float value after appying kernel to x and z
    """
    normed = numpy.linalg.norm(x - z)
    calc = (1/(h**2))*(normed**2)
    return numpy.exp(-calc)
    
def predict_kernel_regression(X, X_train, y_train, kernel_function, h=0.5):
    """ Predict the output values y for the given input design matrix X.

        X: Input matrix in NxM numpy ndarray, where we want to predict the output
            for the vector in each row of X.
        X_train: Design matrix of training input in N_train-x-M numpy ndarray
        y_train: Training output in N_train-x-1 numpy array
        kernel_function: Function that takes two arguments that are each
            Mx1 numpy ndarrays and returns a float value.
        lamb: float value of regularization hyperparameter, lambda (Note, this is a 
            different hyperparameter than the hyperparameter used in RBF kernels)

        Returns: Nx1 numpy ndarray, where the i-th entry is the predicted value 
            corresponding the i-th row vector in X
    """
    (n, m) = X.shape
    (n_train, _) = X_train.shape
    ret = numpy.eye(n, 1)
    for row in range(n):
        total_sum = 0
        for i in range(n_train):
            kern_val = kernel_function(X[row], X_train[i], h)
            total_sum += kern_val

        sum_f = 0
        for i in range(n_train):
            kern_val = kernel_function(X[row], X_train[i], h)
            if total_sum == 0:
                w_i = kern_val
            else:
                w_i = kern_val / total_sum
            sum_f += (w_i * y_train[i])

        ret[row] = sum_f

    return ret

def mse(y, y_hat):
    err = y - y_hat
    sqerr = err**2
    return np.mean(sqerr)

