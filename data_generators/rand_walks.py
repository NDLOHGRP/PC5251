import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def gen_random_walk(n_steps, n_dims=1):
    """
    Returns the trajectory of n_dims-dimensional 
    random walker that takes n_steps random steps
    """
    #np.random.rand returns random numbers between the interval (0,1)
    steps = np.random.rand(n_steps, n_dims) - 0.5
    return np.cumsum(steps, axis=0)


def gen_noisy_line(n_meas, std=50., k=3., offset=0., show_plot=False, show_solution=False):
    """
    Generates measurements from a linear relationship
    y = k x + offset + noise
    where noise is normally distributed N(0, std)
    """
    x_vals = 100.*np.random.rand(n_meas)
    noise = np.random.normal(0., std, n_meas)
    y_vals = k*x_vals + offset + noise
    if show_plot:
        plt.plot(x_vals, y_vals, 'k.', label='measurements')
        plt.xlabel('x inputs')
        plt.ylabel('noisy F measurements')
        if show_solution:
            plt.plot(x_vals, k*x_vals + offset, 'r-', label='ground truth')
        plt.legend()
    return (x_vals, y_vals)

def gen_noisy_curve(n_meas, std=50., k=3., offset=0., show_plot=False, show_solution=False):
    """
    Generates measurements from a linear relationship
    y = k x + offset + noise
    where noise is normally distributed N(0, std)
    """
    x_vals = 100.*np.random.rand(n_meas)
    noise = np.random.normal(0., std, n_meas)
    y_vals = k*np.sin(x_vals/10.) + offset + noise
    if show_plot:
        plt.plot(x_vals, y_vals, 'k.', label='measurements')
        plt.xlabel('x inputs')
        plt.ylabel('noisy F measurements')
        if show_solution:
            plt.plot(x_vals, k*np.sin(x_vals/10.) + offset, 'r.', label='ground truth')
        plt.legend()
    return (x_vals, y_vals)

def plot_1D_random_walker(traj, title=None):
    plt.plot(traj)
    plt.xlabel('steps')
    plt.ylabel('position')
    if title is not None: plt.title(title)
    
def plot_2D_random_walker(traj, showlines=False, title=None):
    if showlines:
        plt.plot(traj[:,0], traj[:,1], 'k-.')
    else:
        plt.plot(traj[:,0], traj[:,1], 'k,')
    plt.plot(0,0, 'ro', markersize=10, label='start')
    plt.xlabel('x position')
    plt.ylabel('y position')
    if title is not None: plt.title(title)
    plt.legend()
    
def plot_1D_histogram(traj, n_bins=100, title=None, logscale=True):
    H = plt.hist(traj, bins=n_bins)
    plt.xlabel('x position')
    plt.ylabel('occurences')
    if title is not None: plt.title(title)
    if logscale: plt.yscale('log')
    return H

def plot_2D_histogram(traj, n_bins=20, title=None):
    fig = plt.figure(figsize=(6,6))

    #We use ImageGrid to make the colorbar formatting easier..
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1),
                         axes_pad=0.05,
                         label_mode="L"
                         )
    H = grid[0].hist2d(traj[:,0], traj[:,1], bins=n_bins)
    grid[0].plot(0,0, 'ro', markersize=5, label='start')
    extent = [H[1].min(), H[1].max(), H[2].min(), H[2].max()]
    ct = grid[0].contour(H[0], 3, extent=extent, colors='black')
    grid[0].clabel(ct, inline=1, fontsize=10)
    grid[0].set_xlabel('x position')
    grid[0].set_ylabel('y position')
    if title is not None: grid[0].set_title(title)
    grid[0].legend()
    return H
    
def plot_1D_log_likelihood(hypo_x, log_likelihoods, 
                           ylabel=None, xlabel=None, title=None):
    """
    It's useful to have a re-usable and flexible plotting function
    to visualize the results of a 1D maximum-likelihood grid search 
    """
    [min_log, max_log] = [log_likelihoods.min(), log_likelihoods.max()]
    plt.plot(hypo_x, log_likelihoods)
    max_loc = np.argmax(log_likelihoods)
    most_likely_x, max_likelihood = [hypo_x[max_loc], log_likelihoods[max_loc]]
    plt.vlines(most_likely_x, min_log, max_log, colors='black', 
           label="most likely:{:0.3f}".format(most_likely_x))
    plt.legend()
    
    if xlabel is None:
        plt.xlabel('hypothetical x_vals')
    else:
        plt.xlabel(xlabel)
        
    if ylabel is None:
        plt.ylabel('log_likelihood')
    else:
        plt.xlabel(ylabel)
    
    if title is not None: 
        plt.title(title)
    return (most_likely_x, max_likelihood)
    
def gen_gaussian(x_vals, y_max, mean, std_dev):
    """
    Returns vertically-rescaled Gaussian PDF at x_vals inputs  
    """
    return y_max * np.exp(-0.5*(np.fabs(x_vals - mean)/std_dev)**2)

def log_gaussian(x_vals, mean, std_dev):
    """
    Returns logarithm of Gaussian likelihood at x_vals inputs
    """
    t0 = -0.5*np.log(2.*np.pi) - np.log(std_dev)
    t1 = -0.5*(np.fabs(x_vals - mean)/std_dev)**2
    return t0 + t1


def find_MLE_mean_std_1D_gaussian(in_vals, n_mean=200, n_std=200, show_plot=True):
    """
    Fancier grid search (a.k.a 'brute-force') for most likely 
    mean and standard deviation simultaneously.    
    """
    mean_guess = np.mean(in_vals)
    std_dev_guess = np.std(in_vals)

    hypo_mean = np.linspace(-0.5+mean_guess, 0.5+mean_guess, n_mean)
    hypo_std_dev = np.linspace(std_dev_guess*0.95, std_dev_guess*1.05, n_std)
    
    log_likelihoods = np.zeros((n_mean, n_std))
    for n,hm in enumerate(hypo_mean):
        log_likelihoods[n] = np.asarray([log_gaussian(in_vals, hm, hsd).mean() for hsd in hypo_std_dev])
    
    #Extract the row,column index of a 2D argmax 
    ind = np.unravel_index(np.argmax(log_likelihoods, axis=None), log_likelihoods.shape)
    mle_mean, mle_std_dev = [hypo_mean[ind[0]], hypo_std_dev[ind[1]]]
    
    if show_plot:
        fig = plt.figure(figsize=(6,6))

        #We use ImageGrid to make the colorbar formatting easier..
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 1),
                         axes_pad=0.05,
                         label_mode="L",
                         cbar_location="right",
                         cbar_mode="single"
                         )
        #This extent is for plotting imshow with the appropriate ticks 
        extent = [hypo_std_dev.min(), hypo_std_dev.max(),hypo_mean.min(), hypo_mean.max()]
        plt_aspect = (extent[1]-extent[0])/(extent[3]-extent[2])

        im = grid[0].imshow(log_likelihoods, origin='lower', extent=extent, aspect=plt_aspect)
        ct = grid[0].contour(log_likelihoods, extent=extent, colors='black')
        grid[0].clabel(ct, inline=1, fontsize=10)

        grid[0].plot(mle_std_dev,mle_mean,'kx', markersize=20, 
                 label=r'most likely [$\mu$, $\sigma$]: [{:0.3f}, {:0.3f}]'.format(mle_mean, mle_std_dev))
        grid.cbar_axes[0].colorbar(im)
        grid[0].set_title('log_likelihood')
        grid[0].set_ylabel('hypothetical means')
        grid[0].set_xlabel('hypothetical std_dev')
        grid[0].legend()
    return (mle_mean, mle_std_dev)

