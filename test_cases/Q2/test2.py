import numpy as np
import matplotlib.pyplot as plt
import os

import util

import kernel_regression

def max_score():
    return 8

def timeout():
    return 300

def test():

    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    expected_mse_train_values = {}
    expected_mse_val_values = {}

    expected_mse_train_values[(1, 0.5)] = 0.0
    expected_mse_val_values[(1, 0.5)] = 12.395160789365237
    expected_mse_train_values[(1, 1)] = 0.0
    expected_mse_val_values[(1, 1)] = 8.660258124069864
    expected_mse_train_values[(1, 2)] = 0.0
    expected_mse_val_values[(1, 2)] = 8.660258124069864
    expected_mse_train_values[(2, 0.5)] = 0.0
    expected_mse_val_values[(2, 0.5)] = 7.888257844693524
    expected_mse_train_values[(2, 1)] = 0.0
    expected_mse_val_values[(2, 1)] = 7.886702594532336
    expected_mse_train_values[(2, 2)] = 3.49579451877525e-22
    expected_mse_val_values[(2, 2)] = 7.860691251264413
    expected_mse_train_values[(10, 0.5)] = 2.481841649215936e-20
    expected_mse_val_values[(10, 0.5)] = 4.568324468316334
    expected_mse_train_values[(10, 1)] = 1.6425050598110614e-05
    expected_mse_val_values[(10, 1)] = 4.573467428920369
    expected_mse_train_values[(10, 2)] = 0.12650998237410754
    expected_mse_val_values[(10, 2)] = 4.883405030426682
    expected_mse_train_values[(50, 0.5)] = 0.0040256460646949125
    expected_mse_val_values[(50, 0.5)] = 1.9608926912965352
    expected_mse_train_values[(50, 1)] = 0.045933624570029245
    expected_mse_val_values[(50, 1)] = 1.9766752540055719
    expected_mse_train_values[(50, 2)] = 0.5174982643081784
    expected_mse_val_values[(50, 2)] = 3.2137548334896673
    expected_mse_train_values[(200, 0.5)] = 0.011746659861276708
    expected_mse_val_values[(200, 0.5)] = 0.11899883462367711
    expected_mse_train_values[(200, 1)] = 0.09890285440150805
    expected_mse_val_values[(200, 1)] = 0.20221300533991166
    expected_mse_train_values[(200, 2)] = 0.6951787119785006
    expected_mse_val_values[(200, 2)] = 1.3704716714660687

    X_train, y_train, X_val, y_val = kernel_regression.load_data()

    for N in [1, 2, 10, 50, 200]:
        for h in [0.5, 1, 2]:
            print('Testing RBF kernel for N={} h={}'.format(N, h))

            f = lambda X: kernel_regression.predict_kernel_regression(X, X_train[:N], y_train[:N], kernel_regression.kernel_rbf, h)

            # Compute and check training error
            mse_train = kernel_regression.mse(y_train[:N], f(X_train[:N]))

            print('\tmse_train:  {:.4f}'.format(mse_train))
            expected_mse_train = expected_mse_train_values[(N, h)]
            threshold = 1e-11 if expected_mse_train < 1e-7 else 0.01
            assert abs(mse_train - expected_mse_train) < threshold, 'Incorrect training MSE value found for RBF kernel N={}, h={}. Expected {}, found {}'.format(N, h, expected_mse_train, mse_train)

            # Compute and check validation error
            mse_val = kernel_regression.mse(y_val, f(X_val))

            print('\tmse_val:    {:.4f}'.format(mse_val))
            expected_mse_val = expected_mse_val_values[(N, h)]
            assert abs(mse_val - expected_mse_val) < 0.01 , 'Incorrect validation MSE value found for RBF kernel N={}, h={}. Expected {}, found {}'.format(N, h, expected_mse_val, mse_val)

            # Plot training data and predicted surface

            # filename = '{}/regression_rbf_N_{}_h_{:0.2f}.png'.format(figures_directory, N, h)
            # filename = filename.replace('.', '_', 1)
            # title = 'RBF kernel, h = {}, N = {}'.format(h, N)

            # points = np.hstack((X_train[:N], y_train[:N]))

            # util.plot_surface(f, points, title=title,
            #         new_figure=True, show_figure=False, save_filename=filename)
            # plt.close()
    
    """
    This section of the code generates the plots that you should put in the results for question 4.
    """
    np.random.seed(0)
    from matplotlib import cm
    f_lin = lambda x, y: 2*x + 3*y
    f_nonlin = lambda x, y: y*(3**(np.sin(y*2)/2))*np.cos(x/10) 
    x_coords = np.random.permutation(np.linspace(0, 10, 1000))
    y_coords = np.random.permutation(np.linspace(0, 10, 1000))
    z_lin_without_noise = [3*x_coords[i] + 3*y_coords[i] for i in range(len(x_coords))]
    z_lin_with_noise = z_lin_without_noise + np.random.normal(0, 2, (1000,))
    z_nonlin_without_noise = [y_coords[i]*(3**(np.sin(y_coords[i]*2)/2))*np.cos(x_coords[i]/10) for i in range(len(x_coords))]
    z_nonlin_with_noise = z_nonlin_without_noise + np.random.normal(0, 2, (1000,))
    X_data = np.concatenate([np.expand_dims(x_coords, axis = 1), np.expand_dims(y_coords, axis = 1)], axis = 1)
    X_train = X_data[:600]
    X_test = X_data[600:]
    y_train = z_lin_with_noise[:600]

    # linear 0.5
    X, Y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    Z = f_lin(X, Y)
    box_preds_lin = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_boxcar, h = 0.5)
    rbf_preds_lin = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_rbf, h = 0.5)
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 0.7)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], box_preds_lin, color='tab:red', label = 'test data points')
    plt.savefig('boxcar_linear_predictions.png')
    plt.close()

    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], rbf_preds_lin, color='tab:red', label = 'test data points')
    plt.savefig('rbf_linear_predictions.png')
    plt.close()

    # nonlinear boxcar [1., 2.]
    y_train = z_nonlin_with_noise[:600]
    # do predictions
    box_preds_05 = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_boxcar, h = 1)
    box_preds_1 = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_boxcar, h = 2)
    X, Y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    Z = f_nonlin(X, Y)
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], box_preds_05, color='tab:red', label = 'test data points')
    plt.savefig('box_nonlinear_h_1.png')
    plt.close()
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], box_preds_1, color='tab:red', label = 'test data points')
    plt.savefig('box_nonlinear_h_2.png')
    plt.close()

    # nonlinear rbf [0.5, 1., 2.]
    rbf_preds_05 = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_rbf, h = 0.5)
    rbf_preds_1 = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_rbf, h = 1)
    rbf_preds_2 = kernel_regression.predict_kernel_regression(X_test, X_train, y_train, kernel_regression.kernel_rbf, h = 2)
    X, Y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    Z = f_nonlin(X, Y)
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], rbf_preds_05, color='tab:red', label = 'test data points')
    plt.savefig('rbf_nonlinear_h_05.png')
    plt.close()
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], rbf_preds_1, color='tab:red', label = 'test data points')
    plt.savefig('rbf_nonlinear_h_1.png')
    plt.close()
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(10, -10)
    ax.plot_surface(X, Y, Z, color = 'tab:green', alpha = 1.0)
    ax.scatter3D(X_test[:, 0], X_test[:, 1], rbf_preds_2, color='tab:red', label = 'test data points')
    plt.savefig('rbf_nonlinear_h_2.png')
    plt.close()

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

if __name__ == "__main__":
    test()
