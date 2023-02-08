import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_marker_size():
    return 20

def get_marker_edge_width():
    return 2

def get_line_width():
    return 2

def get_font_size():
    return 24

def get_tick_font_size():
    return 18

def plot_surface(f, points=None, x_min=-10, x_max=10, z_min=0, z_max=20,
        title=None, new_figure=True, show_figure=True, save_filename=None):
    """ Plot surface of f as well as points if they are not None.

        f: Function that takes as input an Nx2 numpy ndarray and outputs an 
            Nx1 numpy ndarry
        points (default=None): Nx3 numpy ndarray that contains N rows of points, [x_1, x_2, y]
        x_min: Minimum of input range for x_1 (x-axis) and x_2 (y-axis)
        x_max: Maximum of input range for x_1 (x-axis) and x_2 (y-axis)
        z_min: Minimum of output range (z-axis)
        z_max: Maximum of output range (z-axis)
        title (default=None): Title of plot if not None
        new_figure (default=True): If true, calls plt.figure(), which create a 
            figure. If false, it will modify an existing figure (if one exists).
        show_figure (default=True): If true, calls plt.show(), which will open
            a new window and block program execution until that window is closed
        save_filename (defalut=None): If not None, save figure to save_filename 
    """

    N1 = 101
    N = N1**2
    
    x1 = np.linspace(x_min, x_max, N1)
    x2 = np.linspace(x_min, x_max, N1)
    X1, X2 = np.meshgrid(x1,x2)

    X1_flat = X1.reshape((N,1))
    X2_flat = X2.reshape((N,1))
    X = np.hstack((X1_flat, X2_flat))
    Y = f(X).reshape((N1, N1))
    
    if new_figure:
        plt.figure(figsize=(12,8))

    ax = plt.gca(projection='3d')
    ax.plot_surface(X1, X2, Y, alpha=0.3, rstride=2, cstride=2, linewidth=3, color='tab:green', zorder=2)
    ax.plot_wireframe(X1, X2, Y, rstride=10, cstride=10, color='tab:green', zorder=2)

    if points is not None:
        f_points = f(points[:,:2]).flatten()
        points_above = points[np.where(points[:,2]>=f_points)]
        points_below = points[np.where(points[:,2]<f_points)]
        if len(points_below) > 0:
            ax.plot(points_below[:,0], points_below[:,1], points_below[:,2], '.', color='tab:blue', markersize=get_marker_size(), zorder=1)
        if len(points_above) > 0:
            ax.plot(points_above[:,0], points_above[:,1], points_above[:,2], '.', color='tab:blue', markersize=get_marker_size(), zorder=10)
    
    ax.set_xlabel('x_1', fontsize=get_font_size(), labelpad=15)
    ax.set_ylabel('x_2', fontsize=get_font_size(), labelpad=15)
    ax.set_zlabel('y', fontsize=get_font_size(), labelpad=15)
    ax.tick_params(axis='x', labelsize=get_tick_font_size())
    ax.tick_params(axis='y', labelsize=get_tick_font_size())
    ax.tick_params(axis='z', labelsize=get_tick_font_size())
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=6)
    ax.locator_params(axis='z', nbins=6)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_zlim(z_min, z_max)

    ax.view_init(30, 235)

    if title is not None:
        plt.title(title, fontsize=get_font_size())

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()


