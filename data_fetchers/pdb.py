from collections import Counter
import numpy as np
import glob
import urllib.request 
from sklearn.feature_extraction.text import CountVectorizer

class pdb_parser:
    def __init__(self):
        self.res_types = {"charged":['ARG', 'LYS','ASP', 'GLU'],
       "polar":['GLN', 'ASN', 'HIS', 'SER', 'THR', 'TYR', 'CYS', 'TRP'],
       "hydrophobic":['ALA', 'ILE','LEU', 'MET','PHE', 'VAL', 'PRO', 'GLY']}
        self.res_types_dict = {}
        for aa_type, aa_labels in self.res_types.items():
            for label in aa_labels:
                self.res_types_dict[label] = aa_type
        self.pdb_rows = None
        self.chain_seq = None
        self.unique_chains = None
        
    def remap_residue(self, s):
        map_dict = {'ARG':'R', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'GLN':'Q', 'ASN':'N', 'HIS':'H', 
                    'SER':'S', 'THR':'T', 'TYR':'Y', 'CYS':'C', 'CYS':'C', 'TRP':'W', 'ALA':'A', 
                    'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'VAL':'V', 'PRO':'P', 'GLY':'G'}
        return [map_dict[r] for r in s]

    def fetch_pdb(self,filename):
        """
        Checks if pdb filename has already been downloaded. 
        If not, fetches it from PDB databank.
        Then reads the lines of the pdb file.
        """
        downloaded_pdb_files = glob.glob("*.pdb")
        if filename in downloaded_pdb_files:
            print(filename + " already downloaded")
        else:
            url = 'https://files.rcsb.org/download/' + filename
            print("Getting missing file from..." + url)
            urllib.request.urlretrieve(url, filename)
        
        fp = open(filename)    #Create a file-access object pointing to the file's contents
        self.pdb_rows = fp.readlines() #Read in the lines of this file, interpreted as text, into a list
        fp.close()             #Close the file-access object
        
    def fetch_chain_ID(self, row):
        """
        Residue sequences only begin from character 19 onwards
        """
        return row[11]

    def fetch_residues(self, row):
        """
        Residue sequences only begin from character 19 onwards
        """
        return row[19:].split()

    def extract_seqres(self, in_func):
        """
        Here we can pass a function (in_func) to a 'template' function (extract_seqres) as argument.
        This allows you to reuse this 'template' function.
        Another way to accomplish such template functions is by using Python decorators. 
        """
        return [in_func(x) for x in self.pdb_rows if x[:6]=='SEQRES']

    def extract_chains_from_pdb(self, filename):
        """
        You are supposed to implement this using the functions above.
        Input: 
            a PDB filename (e.g. 1aus.pdb), 
        Output: 
            returns a list of residues described within the PDB file.
        """

        self.fetch_pdb(filename)
        lst_of_res_lst = [x for x in self.extract_seqres(self.fetch_residues)]

        #Extract the chain_ID of each row into a list (as many elements in list as SEQRES lines)  
        lst_of_chainID_lst = [x for x in self.extract_seqres(self.fetch_chain_ID)]
        unique_counter = Counter(lst_of_chainID_lst)

        #Creates a dictionary with key:value of chain_ID:chain_sequence 
        self.chain_seq = {k:[] for k in set(lst_of_chainID_lst)}
        for chain, seq in zip(lst_of_chainID_lst, lst_of_res_lst):
            self.chain_seq[chain]+= seq
            
        self.unique_seq = list(set([' '.join(cs) for cs in self.chain_seq.values()]))
            
    def compute_entropy_of_chains(self, chains_as_sentences):
        """
        Here we compute the entropy of input residue chains. 
        Parameters
        ----------
        chains_as_sentences : list of strings
            List of strings, where each string is a concatenation of each residue chain 
            in the pdb file (e.g., ["MET PRO CYS ...", "MET ARG ALA ..."]).
        """
        corpus = list(chains_as_sentences)

        # Recasts sentences as a bag of words.
        vectorizer = CountVectorizer(lowercase=False)
        output = vectorizer.fit_transform(corpus)

        #Features here represent the words, or the residues in your polypetide chain
        features = np.asarray(vectorizer.get_feature_names())

        #Each sentence is now described only by the frequency that each word in the bag occurs. 
        #Notice we've added 1. to all frequencies below. This is known as the Dirichlet prior.
        frequencies = output.toarray()

        #Normalize the frequencies to sum to 1
        tot_freq = frequencies.sum(axis=1)
        frequencies = frequencies/tot_freq[:,None]

        #Computing the log2 entropies are just as simple as applying the formula.
        #Avoid zeros with threshold
        thres = np.finfo(float).resolution
        chain_entropies = [(-x[x>thres]*np.log2(x[x>thres])).sum() for x in frequencies]

        return chain_entropies
    
    def compute_entropy_of_each_chain(self):
        """
        Computes entropy of each residue chain in protein assembly.
        """
        chains_as_sentences = [' '.join(cs) for cs in self.chain_seq.values()]
        return self.compute_entropy_of_chains(chains_as_sentences)
    
    def compute_entropy_of_each_unique_chain(self):
        """
        Computes entropy of each unique residue chain in protein assembly.
        """
        chains_as_sentences = list(set([' '.join(cs) for cs in self.chain_seq.values()]))
        return self.compute_entropy_of_chains(chains_as_sentences)

    def compute_average_entropy_of_all_unique_chains(self):
        """
        Computes average entropy of all unique residue chains in protein assembly.
        """
        return np.mean(self.compute_entropy_of_each_unique_chain())
    
    def compute_average_entropy_of_all_chains(self):
        """
        Computes average entropy of all residue chains in protein assembly.
        """
        return np.mean(self.compute_entropy_of_each_chain())


