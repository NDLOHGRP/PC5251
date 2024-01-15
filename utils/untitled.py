import numpy as np

def my_cov(input_design_matrix):
    res = np.cov(input_design_matrix)
    return res

def my_pearson_from_cov(input_covariance_matrix):
    ...
    return


c, design_matrix = rc.gen_random_correlated_design_matrix(rand_seed = 1234578)

#First test
res1 = my_cov(design_matrix)
np.allclose(res1, np.cov(design_matrix))

#Second test
res2 = my_pearson_from_cov(res1)
np.allclose(res2, np.corrcoef(design_matrix))