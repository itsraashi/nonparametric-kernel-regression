import numpy as np
import matplotlib.pyplot as plt
import os

import util

import kernel_regression

def max_score():
    return 7

def timeout():
    return 300

def test():
    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    expected_mse_train_values = {}
    expected_mse_val_values = {}

    expected_mse_train_values[(1, 0.5)] = 0.0
    expected_mse_val_values[(1, 0.5)] = 107.78211802288456
    expected_mse_train_values[(1, 1)] = 0.0
    expected_mse_val_values[(1, 1)] = 107.78211802288456
    expected_mse_train_values[(1, 2)] = 0.0
    expected_mse_val_values[(1, 2)] = 107.78211802288456
    expected_mse_train_values[(2, 0.5)] = 0.0
    expected_mse_val_values[(2, 0.5)] = 107.78211802288456
    expected_mse_train_values[(2, 1)] = 0.0
    expected_mse_val_values[(2, 1)] = 107.78211802288456
    expected_mse_train_values[(2, 2)] = 0.0
    expected_mse_val_values[(2, 2)] = 107.78211802288456
    expected_mse_train_values[(10, 0.5)] = 0.0
    expected_mse_val_values[(10, 0.5)] = 107.78211802288456
    expected_mse_train_values[(10, 1)] = 0.0
    expected_mse_val_values[(10, 1)] = 107.78211802288456
    expected_mse_train_values[(10, 2)] = 0.0
    expected_mse_val_values[(10, 2)] = 107.78211802288456
    expected_mse_train_values[(50, 0.5)] = 0.0
    expected_mse_val_values[(50, 0.5)] = 107.78211802288456
    expected_mse_train_values[(50, 1)] = 0.009451693822817306
    expected_mse_val_values[(50, 1)] = 78.50573030646987
    expected_mse_train_values[(50, 2)] = 0.030146409094203745
    expected_mse_val_values[(50, 2)] = 56.64142201152809
    expected_mse_train_values[(200, 0.5)] = 0.0005934072672168511
    expected_mse_val_values[(200, 0.5)] = 96.41540136905572
    expected_mse_train_values[(200, 1)] = 0.018836242817364682
    expected_mse_val_values[(200, 1)] = 49.03832199172827
    expected_mse_train_values[(200, 2)] = 0.109652925453873
    expected_mse_val_values[(200, 2)] = 0.05344278858444126

    X_train, y_train, X_val, y_val = kernel_regression.load_data()

    for N in [1, 2, 10, 50, 200]:
        for h in [0.5, 1, 2]:
            print('Testing boxcar kernel for N={} h={}'.format(N, h))

            f = lambda X: kernel_regression.predict_kernel_regression(X, X_train[:N], y_train[:N], kernel_regression.kernel_boxcar, h)

            # Compute and check training error
            mse_train = kernel_regression.mse(y_train[:N], f(X_train[:N]))

            print('\tmse_train:  {:.4f}'.format(mse_train))
            expected_mse_train = expected_mse_train_values[(N, h)]
            threshold = 1e-11 if expected_mse_train < 1e-7 else 0.01
            assert abs(mse_train - expected_mse_train) < threshold, 'Incorrect training MSE value found for boxcar kernel N={}, h={}. Expected {}, found {}'.format(N, h, expected_mse_train, mse_train)

            # Compute and check validation error
            mse_val = kernel_regression.mse(y_val, f(X_val))

            print('\tmse_val:    {:.4f}'.format(mse_val))
            expected_mse_val = expected_mse_val_values[(N, h)]
            assert abs(mse_val - expected_mse_val) < 0.01 , 'Incorrect validation MSE value found for boxcar kernel N={}, h={}. Expected {}, found {}'.format(N, h, expected_mse_val, mse_val)

            # Plot training data and predicted surface

            # filename = '{}/regression_boxcar_N_{}_h_{:0.2f}.png'.format(figures_directory, N, h)
            # filename = filename.replace('.', '_', 1)
            # title = 'Boxcar kernel, h = {}, N = {}'.format(h, N)

            # points = np.hstack((X_train[:N], y_train[:N]))

            # util.plot_surface(f, points, title=title,
            #         new_figure=True, show_figure=False, save_filename=filename)
            # plt.close()
            

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

if __name__ == "__main__":
    test()
