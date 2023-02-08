import numpy as np
import matplotlib.pyplot as plt
import os

import util

import kernel_regression

def max_score():
    return 10

def timeout():
    return 60

def test():

    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    expected_output_values = {}

    expected_output_values[(1.0, (0, 0), (0, 0))] = 1.0
    expected_output_values[(1.0, (0, 0), (0, -2))] = 0.01831563888873418
    expected_output_values[(1.0, (0, 0), (-0.5, -0.5))] = 0.6065306597126333
    expected_output_values[(1.0, (0, 0), (-0.1, 0.2))] = 0.951229424500714
    expected_output_values[(1.0, (10, 10), (10, 10))] = 1.0
    expected_output_values[(1.0, (10, 10), (10, 8))] = 0.01831563888873418
    expected_output_values[(1.0, (10, 10), (9.5, 9.5))] = 0.6065306597126333
    expected_output_values[(1.0, (10, 10), (9.9, 10.2))] = 0.9512294245007143
    expected_output_values[(1.0, (2, -3), (2, -3))] = 1.0
    expected_output_values[(1.0, (2, -3), (2, -5))] = 0.01831563888873418
    expected_output_values[(1.0, (2, -3), (1.5, -3.5))] = 0.6065306597126333
    expected_output_values[(1.0, (2, -3), (1.9, -2.8))] = 0.9512294245007139
    expected_output_values[(0.5, (0, 0), (0, 0))] = 1.0
    expected_output_values[(0.5, (0, 0), (0, -2))] = 1.1253517471925912e-07
    expected_output_values[(0.5, (0, 0), (-0.5, -0.5))] = 0.13533528323661262
    expected_output_values[(0.5, (0, 0), (-0.1, 0.2))] = 0.8187307530779818
    expected_output_values[(0.5, (10, 10), (10, 10))] = 1.0
    expected_output_values[(0.5, (10, 10), (10, 8))] = 1.1253517471925912e-07
    expected_output_values[(0.5, (10, 10), (9.5, 9.5))] = 0.13533528323661262
    expected_output_values[(0.5, (10, 10), (9.9, 10.2))] = 0.818730753077983
    expected_output_values[(0.5, (2, -3), (2, -3))] = 1.0
    expected_output_values[(0.5, (2, -3), (2, -5))] = 1.1253517471925912e-07
    expected_output_values[(0.5, (2, -3), (1.5, -3.5))] = 0.13533528323661262
    expected_output_values[(0.5, (2, -3), (1.9, -2.8))] = 0.8187307530779816
    expected_output_values[(2.0, (0, 0), (0, 0))] = 1.0
    expected_output_values[(2.0, (0, 0), (0, -2))] = 0.36787944117144233
    expected_output_values[(2.0, (0, 0), (-0.5, -0.5))] = 0.8824969025845953
    expected_output_values[(2.0, (0, 0), (-0.1, 0.2))] = 0.9875778004938814
    expected_output_values[(2.0, (10, 10), (10, 10))] = 1.0
    expected_output_values[(2.0, (10, 10), (10, 8))] = 0.36787944117144233
    expected_output_values[(2.0, (10, 10), (9.5, 9.5))] = 0.8824969025845953
    expected_output_values[(2.0, (10, 10), (9.9, 10.2))] = 0.9875778004938816
    expected_output_values[(2.0, (2, -3), (2, -3))] = 1.0
    expected_output_values[(2.0, (2, -3), (2, -5))] = 0.36787944117144233
    expected_output_values[(2.0, (2, -3), (1.5, -3.5))] = 0.8824969025845953
    expected_output_values[(2.0, (2, -3), (1.9, -2.8))] = 0.9875778004938814

    x = np.zeros((2, 1))
    z = np.zeros((2, 1))
    for h in [0.5, 1, 2]:
        for z_tuple in [(0, 0), (10, 10), (2, -3)]:
            z[0, 0] = z_tuple[0]
            z[1, 0] = z_tuple[1]

            x_tuples = [ (z_tuple[0] + 0, z_tuple[1] + 0),
                    (z_tuple[0] + 0, z_tuple[1] - 2),
                    (z_tuple[0] -0.5, z_tuple[1] - 0.5),
                    (z_tuple[0] -0.1, z_tuple[1] + 0.2) ]

            for x_tuple in x_tuples:
                x[0, 0] = x_tuple[0]
                x[1, 0] = x_tuple[1]

                actual_output = kernel_regression.kernel_rbf(x, z, h=h)

                if isinstance(actual_output, np.ndarray):
                    if len(actual_output.shape) == 1:
                        actual_output = actual_output[0]
                    else:
                        actual_output = actual_output[0, 0]

                expected_output = expected_output_values[(h, z_tuple, x_tuple)]
                assert abs(actual_output - expected_output) < 0.001 , 'Incorrect kernel value found for h={}, x={}, z={}. Expected {}, found {}'.format(h, x_tuple, z_tuple, expected_output, actual_output)
     
    test_score = max_score()
    test_output = 'PASS\n'

    # Plot surface of kernel at a specific point z

    k = lambda X: rbf_kernel_all(X, z, 0.5)

    # filename = '{}/kernel_rbf_h_{:0.2f}.png'.format(figures_directory, 0.5)
    # filename = filename.replace('.', '_', 1)
    # title = 'RBF kernel, h = {}, z = ({}, {})'.format(0.5, z[0, 0], z[1, 0])

    # point = np.hstack((z.T, np.ones((1,1))))

    # util.plot_surface(k, point, x_min=-6, x_max=6, z_min=0, z_max=2,
    #         title=title, new_figure=True, show_figure=False, save_filename=filename)
    # plt.close()

    return test_score, test_output

def rbf_kernel_all(X, z, h):
    N, M = X.shape

    y = np.zeros((N, 1))
    for n in range(N):
        x_n = np.reshape(X[n], (M, 1))
        
        y[n] = kernel_regression.kernel_rbf(x_n, z, h=h)

    return y

if __name__ == "__main__":
    test()
