import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from functools import partial 
from time import time

class metropolis_ising:
    def __init__(self, arr_len=20, temp = 1., 
                 field_str=0., block_rad=1, 
                 free_energy_update=5,
                 rand_seed=None):
        """
        Initializes a periodically connected 2D square array of spins whose shape is (arr_len,arr_len).
        
        Parameters
        -----
        
        """
        self.arr_len   = arr_len
        self.num_spins = arr_len*arr_len
        self.temp      = temp
        self.field_str = field_str
        self.epoch_num = 0
        self.rand_seed = rand_seed
        
        self.neighbor_shift_list = [[-1, 0],[0, -1], [0, 1], [1, 0]]
        self.num_neighbors = len(self.neighbor_shift_list)
        
        #To store spin arrays are regular intervals
        self.spin_arr                 = None
        self.spin_arr_memory          = []
        self.spin_arr_epoch_memory    = []
        self.spin_arr_memory_depth    = 25
        
        #Helpful for monitoring state of the system during updates
        self.block_rad                = block_rad
        self.free_energy_update       = free_energy_update
        self.free_energy_epoch_memory = []
        self.flip_energy_entropy      = []
        self.spin_entropy             = []
        self.average_spin             = []
        self.spin_block_entropy       = []
        self.helmholtz_energy         = []
        
    def create_random_periodic_spins_on_grid(self):
        """
        Initialize with equal probability for +1 and -1 spins 
        Approximates the high temperature limit.
        
        Parameters
        ------
        seed: int (optional)
            You can specify a random seed for initializing the random spin array
        """
        if type(self.rand_seed) is int:
            np.random.seed(self.rand_seed)
        spin_arr      = np.random.choice([-1, 1], size=self.num_spins, p=[0.5,0.5])
        self.spin_arr = spin_arr.reshape(self.arr_len, self.arr_len).astype(np.int8).copy()
        np.random.seed()
        
    def create_uniform_periodic_spins_on_grid(self):
        """
        Initialize all spins as +1. 
        This is one of two possible ground states (lowest energy state).
        """
        self.spin_arr = np.ones((self.arr_len, self.arr_len), dtype=np.int8)
    
    def compute_energy_of_spin(self, ind):
        """
        Computes the (energy cost divided by temperature) for a spin
        Energy for spin s_i is:
            E = -ext_field*s_i - s_i* (\sum_j s_j)
            where s_j are the four spins that are north, south, east, west of s_i at position ind.
            
        Note that we have assumed Boltzmann constant kB=1, and magnetic coupling strength J=1. 
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin whose energy we are trying to compute
        """
        curr_spin      = self.spin_arr[ind[0], ind[1]]
        energy_of_spin = self.field_str
        
        for shifts in self.neighbor_shift_list:
            r = (shifts[0]+ind[0])%self.arr_len 
            c = (shifts[1]+ind[1])%self.arr_len 
            energy_of_spin    += self.spin_arr[r, c]
        energy_of_spin *= -curr_spin
        return energy_of_spin
        
        
    def compute_energy_change_for_spin_flip(self, ind):
        """
        Computes the (energy cost divided by temperature) for spin flip using
        Energy for spin s_i is:
            E = -ext_field*s_i - s_i* (\sum_j s_j)
            where s_j are the four spins that are north, south, east, west of s_i at position ind.
            
        Note that we have assumed Boltzmann constant kB=1, and magnetic coupling strength J=1. 
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin that we are trying to flip
        """
        curr_spin           = self.spin_arr[ind[0], ind[1]]
        energy_of_spin_flip = self.field_str
        
        for shifts in self.neighbor_shift_list:
            r = (shifts[0]+ind[0])%self.arr_len 
            c = (shifts[1]+ind[1])%self.arr_len 
            energy_of_spin_flip    += self.spin_arr[r, c]
        energy_of_spin_flip *= (2.*curr_spin)/self.temp    
        
        return energy_of_spin_flip
    
    def attempt_spin_flip(self, ind, cost_thres):
        """
        Attempt to flip a spin at ind = [row,col] using Boltzmann distribution as acceptance function.
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin that we are trying to flip
        """
        energy_of_spin_flip = self.compute_energy_change_for_spin_flip(ind)
        
        #Spin flip only if energy for doing so is negative (E_flipped < E_noflip)
        #else if spin_flip energy>0 only flip if below cost-threshold
        #Instead of probability, we use log-probabilities to prevent over/underflow
        log_prob_of_flip = -1.*energy_of_spin_flip
        #cost_thres       = np.log(np.random.rand())
        
        if (energy_of_spin_flip < 0.):
            self.spin_arr[ind[0], ind[1]] *= -1
        elif (log_prob_of_flip > cost_thres):
            self.spin_arr[ind[0], ind[1]] *= -1
    
    def evolve(self, num_epochs=100, store_freq=None, verbose=False):
        """
        Evolve the dynamics of the 2D Ising spins using MCMC algorithm.
        For each epoch, we sequentially attempt to flip each spin. 
        
        You can uncomment one of the lines below flip random spins non-sequentially. 
        Both sequential and non-sequential spin flip recipes should converge to equilibrium states that are within the same thermodynamic ensemble. 
        
        Parameters
        -----
        num_epochs: integer
            Number of epochs where we will evolve the spin configurations using the MCMC algorithm.
        
        store_freq: integer
            How often (measured in epochs) will store a snapshot of the spin configuration
            
        verbose: bool
            Whether to print additional information about the MCMC evolution. 
        """
        t0 = time()
        for e in range(num_epochs):
            #Seed "random temperature fluctuations" if we need to 
            #repeat a trajectory
            if type(self.rand_seed) is int:
                np.random.seed(self.rand_seed+self.epoch_num)
                cost_thresh_array = np.random.rand(self.num_spins)
                np.random.seed()
            else: 
                cost_thresh_array = np.random.rand(self.num_spins)
            cost_thresh_array = np.log(cost_thresh_array)
            
            for pos,ct in zip(range(self.num_spins), cost_thresh_array):
                pos = np.random.randint(0, self.num_spins)
                flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
                self.attempt_spin_flip(flip_candidate, ct)
            self.epoch_num += 1
            
            #Determine if free energy monitors should be computed+stored during this epoch.
            if (self.epoch_num%self.free_energy_update) == 0:
                self.free_energy_epoch_memory.append(self.epoch_num)
                #self.spin_entropy.append(self.compute_spin_entropy())
                self.average_spin.append(self.compute_average_spin())
                self.spin_block_entropy.append(self.compute_spin_block_entropy())
                self.helmholtz_energy.append(self.compute_helmholtz_energy_per_spin())
                
            #Determine if a snapshot of the spin_arr should be saved in this epoch 
            if store_freq is not None:
                if (e%store_freq) == 0:
                    self.store_spin_arr()
                    self.store_epoch_num()
        if verbose:
            print(f'Time per epoch: {(time()-t0)/num_epochs:0.3f}s')
            
    def compute_flip_energy_map(self):
        """
        Computes the energy cost to flip each spin.
        """
        flip_cost = np.zeros(self.num_spins)
        for pos in range(self.num_spins):
            flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
            flip_cost[pos] = self.compute_energy_change_for_spin_flip(flip_candidate)
        return flip_cost.reshape(self.arr_len, self.arr_len)    
    
    def compute_flip_energy_entropy(self):
        """
        Computes the "entropy" associated with flipping each spin.
        """
        flip_energy_map = self.compute_flip_energy_map()
        c               = Counter(flip_energy_map.flatten())
        probs           = list(c.values())
        tot_prob        = np.sum(probs)
        h               = 0.
        for p in probs:
            prob = p/tot_prob
            h    += -(prob) * np.log2(prob)
        return h
                
    def compute_average_spin(self):
        """
        Return the average magnetization of the current spin configuration
        """
        return self.spin_arr.mean()
    
    def compute_spin_entropy(self):
        """
        Computes the average entropy of single spins.
        Deprecated! Should use extract_spin_blocks instead.
        """
        down_spin_frac = (self.spin_arr == -1).sum() / self.num_spins
        up_spin_frac   = (self.spin_arr == 1).sum() / self.num_spins
        return np.sum([-p*np.log2(p) for p in [down_spin_frac, up_spin_frac] if p > 0.5/self.num_spins])
        
    def extract_spin_blocks(self):
        """
        Extracts all square sub-blocks of (self.block_rad+1)**2 spins from the spin_arr.
        The spin-pattern of each spin-block is stored as strings e.g., "-111-1-1-1".
        
        Note: converting to binary 0bXXX words before doing a counter is slower.
        """
        block_range  = range(-self.block_rad, self.block_rad+1)
        block_shifts = [[x,y] for x in block_range for y in block_range]
        
        #Representing spin blocks as strings
        all_words = []
        for pos in range(self.num_spins):
            ind  = np.unravel_index(pos, (self.arr_len, self.arr_len))
            word = []
            for n,shifts in enumerate(block_shifts):
                r = (shifts[0]+ind[0])%self.arr_len 
                c = (shifts[1]+ind[1])%self.arr_len
                word.append(str(self.spin_arr[r, c]))
            all_words.append(''.join(word))
        return all_words
        
#         Uncomment to experiment with binary representation of spin blocks.
#         Remember to comment out the code block above
#         all_binary_words = []
#         for pos in range(self.num_spins):
#             ind = np.unravel_index(pos, (self.arr_len, self.arr_len))
#             binary_word = 0b0
#             for n,shifts in enumerate(block_shifts):
#                 r = (shifts[0]+ind[0])%self.arr_len 
#                 c = (shifts[1]+ind[1])%self.arr_len
#                 if (self.spin_arr[r, c]>0): 
#                     binary_word |= 1<<n
#             all_binary_words.append(binary_word)
#         return all_binary_words
    
    def compute_spin_block_entropy(self):
        """
        Computes the entropy of the spin blocks harvested from spin_arr.
        This entropy estimates the number of possible spin-block patterns that might occur 
        based on ensemble of spin-blocks from spin_arr. 
        
        The output divides this entropy by the number of spins within each spin-block
        to give us average entropy per spin. 
        
        Note that when self.block_rad = 0, this routine essentially returns only the single spin entropy.
        """
        spin_blocks_as_words = self.extract_spin_blocks()
        c        = Counter(spin_blocks_as_words)
        probs    = list(c.values())
        tot_prob = np.sum(probs)
        h        = 0.
        
        for p in probs:
            prob = p/tot_prob
            h    += -(prob) * np.log2(prob)
        return h/((2*self.block_rad+1)**2)
        
    def compute_mean_energy(self):
        """
        Computes the total spin energy of the spin_arr, then divides by the number of spins in spin_arr.
        Approximates the internal energy U of the spin_arr.
        """
        tot_energy = 0.
        for pos in range(self.num_spins):
            flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
            tot_energy     += self.compute_energy_of_spin(flip_candidate)
        return tot_energy/self.num_spins
    
    def compute_helmholtz_energy_per_spin(self):
        """
        Computes the Helmholtz energy of the spin_arr:
            F = U - T S, 
            where U is the internal energy
            T is the average temperature of the spin_arr
            S is the entropy of the spin_arr computed using spin-blocks as features in counting entropy
        """
        return self.compute_mean_energy() - self.compute_spin_block_entropy()*self.temp

    def store_spin_arr(self):
        """
        Append the spin array to a running list of spin-arrays that have a maximum depth.
        We implement a first-in-first-out (FIFO) queue. 
        """
        if len(self.spin_arr_memory) < self.spin_arr_memory_depth:
            self.spin_arr_memory.append(self.spin_arr.copy())
        else:
            self.spin_arr_memory.pop(0)
            self.spin_arr_memory.append(self.spin_arr.copy())
            
    def store_epoch_num(self):
        """
        We implement a first-in-first-out (FIFO) queue.
        """
        if len(self.spin_arr_epoch_memory) < self.spin_arr_memory_depth:
            self.spin_arr_epoch_memory.append(self.epoch_num)
        else:
            self.spin_arr_epoch_memory.pop(0)
            self.spin_arr_epoch_memory.append(self.epoch_num)
            
    def plot_spins(self, num_rows=5, num_cols=5, figsize=(10,10)):
        """
        Plot the stored spin configurations.
        """
        fig, axes   = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
        shared_args = {"xytext":(-1,-1), "va":'top', "ha":"left", "textcoords":'offset points'}
        for ax,arr,lb in zip(axes.ravel(), self.spin_arr_memory, self.spin_arr_epoch_memory):
            ax.imshow(arr, vmin=-1, vmax=1, cmap=plt.cm.bone_r)
            ax.annotate(f'Epoch {lb:d}', xy=(1,0), color='black',
                        bbox=dict(boxstyle="round", fc="white", ec="black"), **shared_args)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
    def plot_summary(self):
        """
        """
        fig, axes = plt.subplots(1,2, figsize=(15,3))
        axes[0].plot(self.free_energy_epoch_memory, self.spin_block_entropy, label='spin block entropy')
        axes[0].legend()
        axes[0].set_xlabel('epoch number')
        axes[0].set_ylabel('metric')

        axes[1].plot(self.free_energy_epoch_memory, self.helmholtz_energy, label='helmholtz energy per spin')
        axes[1].legend()
        axes[1].set_xlabel('epoch number')
        
    def ising_mean_field_transcendental_objective(self, temp, mean_mag):
        """
        Returns the objective function (transcedental equation) to minimize, 
        which gives us the mean field magnetization in a 2D Ising system. 
        
        Parameters
        ------
        temp: floating point
            temperature of the spin system
            
        mean_mag: floating point
            mean magnetization of the spin system

        """
        return np.fabs(mean_mag - np.tanh(self.num_neighbors*mean_mag/temp))

    def compute_mean_mag(self, temp, x0=0.05):
        """
        Solve the transcendental equation that gives us the 
        mean field magnetization for the Ising system at a particular temperature
        ------
        Parameters:
        x0 : floating point
            Initial guess for magnetization when iteratively solving the transcendental equation
        """
        ising_mean_t = partial(self.ising_mean_field_transcendental_objective, temp)
        res          = minimize(ising_mean_t, x0) 
        return np.abs(res.x)
    