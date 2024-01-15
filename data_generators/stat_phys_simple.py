import numpy as np
import pandas as pd

def log_maxwell_boltzmann_likelihood(T, nu):
    return -1.5*np.log(T) + 2*np.log(nu) - nu**2/T

def maxwell_boltzmann_likelihood(T, nu_vec):
    log_likelihoods = log_maxwell_boltzmann_likelihood(T, nu_vec)
    #Log-sum-exp trick
    prob = np.exp(log_likelihoods - log_likelihoods.max())
    #Normalize probability
    prob /= prob.sum()
    return prob

#This is used to generate test data
def gen_maxwell_boltzmann_nus(T, num_samples=1000, nu_range=[0.1, 50, 100]):
    nus_tmp = np.linspace(*nu_range)
    prob_tmp = maxwell_boltzmann_likelihood(T, nus_tmp)
    rand_cdf = np.random.rand(num_samples)
    rand_nu = np.interp(rand_cdf, np.cumsum(prob_tmp), nus_tmp)
    return rand_nu