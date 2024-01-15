import numpy as np

def gen_random_correlated_design_matrix(num_measurements = 100,
                                        num_features = 50,
                                        num_latent_choices = 3,
                                        sig_level = 10.,
                                        rand_seed = None,
                                        output_latent = False):
    #generate ground truth signals.
    if rand_seed is not None:
        np.random.seed(rand_seed)
    sig = np.random.rand(num_latent_choices, num_features)

    #generate latent choices
    latent_choices = np.random.choice(num_latent_choices, num_measurements)

    #generate design matrix
    design_matrix = np.zeros((num_measurements, num_features))
    for i in range(num_measurements):
        design_matrix[i] = np.random.poisson(sig_level*sig[latent_choices[i]])
    
    np.random.seed()

    if output_latent:
        return (latent_choices, design_matrix)
    else:
        return (None, design_matrix)