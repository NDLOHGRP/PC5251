import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def imshow_aspect(to_plot, ax=None):
    """
    Imshow a 2D array that rescales the output aspect ratio to be num_cols/num_rows.  
    """
    r,c = to_plot.shape
    if ax is None:
        plt.imshow(to_plot, aspect=c/r)
    else:
        ax.imshow(to_plot, aspect=c/r)
        
def imshow_with_pretty_colorbar(input_arr, figsize=(5,5)):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    div = make_axes_locatable(ax)
    im = ax.imshow(input_arr)
    cax = div.append_axes("right", size="7%", pad="3%")
    cb = fig.colorbar(im, cax=cax)
    return ax

def plot_2D_decision_boundaries(model, in_X, in_ax=None, s=1, alpha=0.05, cm=plt.cm.rainbow_r):
    """
    Plots the decision boundaries of a trained model given the training data with 2D features.
    2D feature range of training data (in_X) is used to create a square grid of samples.
    The trained model will return class labels for each of these samples, and these labels will
    be used to plot the decision boundaries.

    Parameters:
    -----
    model: object
        must have model.predict() implemented to return class labels

    in_X: numpy array of shape (N,2)
        training data array.
    """
    r_min, c_min = in_X.min(axis=0)
    r_max, c_max = in_X.max(axis=0)
    num_pts = 300
    r_grid, c_grid = np.mgrid[r_min:r_max:num_pts*1j, c_min:c_max:num_pts*1j]
    samp_points = np.vstack([r_grid.ravel(), c_grid.ravel()]).T
    pred_labels = model.predict(samp_points)
    if in_ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        ax.scatter(samp_points[:,0], samp_points[:,1], s=s, alpha=alpha, c=pred_labels, cmap=cm)
        return ax
    else:
        in_ax.scatter(samp_points[:,0], samp_points[:,1], s=s, alpha=alpha, c=pred_labels, cmap=cm)

def plot_2D_decision_boundaries_tf(model, in_X, in_ax=None, s=1, alpha=0.05, cm=plt.cm.rainbow_r):
    """
    Plots the decision boundaries of a trained model given the training data with 2D features.
    2D feature range of training data (in_X) is used to create a square grid of samples.
    The trained model will return class labels for each of these samples, and these labels will
    be used to plot the decision boundaries.

    Parameters:
    -----
    model: object
        must have model.predict_classes() implemented to return class labels

    in_X: numpy array of shape (N,2)
        training data array.
    """
    r_min, c_min = in_X.min(axis=0)
    r_max, c_max = in_X.max(axis=0)
    num_pts = 300
    r_grid, c_grid = np.mgrid[r_min:r_max:num_pts*1j, c_min:c_max:num_pts*1j]
    samp_points = np.vstack([r_grid.ravel(), c_grid.ravel()]).T
    pred_labels = model.predict_classes(samp_points)
    if in_ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        ax.scatter(samp_points[:,0], samp_points[:,1], s=s, alpha=alpha, c=pred_labels, cmap=cm)
        return ax
    else:
        in_ax.scatter(samp_points[:,0], samp_points[:,1], s=s, alpha=alpha, c=pred_labels, cmap=cm)