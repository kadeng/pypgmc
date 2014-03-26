
from . import DiscretePGM, PotentialTable, CliqueTreeInference, LoopyBPInference
import networkx as nx
import pandas
import numpy as np

class BayesNet(DiscretePGM):
    ''' Implementation of a simple Bayesian Network.'''
    
    def __init__(self, digraph, cardinalities=[], var_names=[]):
        ''' Construct bayesian network from a DiGraph which specifies causal directions of variables 
            Arguments:
                digraph: networkx.DiGraph instance
        '''
        assert isinstance(digraph, nx.DiGraph)
        DiscretePGM.__init__(self, cardinalities, var_names)
        self.G = digraph
        self.value_map = {}
        
    def learn_variable_values(self, var_name, series):
        if (not isinstance(series, pandas.Series)):
            series = pandas.Series(series)
        uvals = sorted(series.unique())
        self.values[var_name] = uvals
        self.add_var(len(uvals), var_name)
         
    def learn_cardinalities(self, data_frame):
        assert isinstance(data_frame, pandas.DataFrame)
        vars = self.G.nodes()
        learned = set()
        for v in vars:
            if (not v in self.var_names and v in data_frame.columns):
                self.learn_variable_values(v, data_frame[v])
                learned.add(v)
        return learned
    
    def init_inference(self, logspace=False, max_ctree_mem=40000000):
        factor_scopes = []
        for v in self.G.nodes():
            in_edges = self.G.in_edges(v)
            parents = [e[0] for e in in_edges]
            vscope = set(parents+ [v])
            factor_scopes.append(vscope)
        inference  = CliqueTreeInference(factor_scopes, self, logspace)
        musage = inference.get_mem_usage()
        if (musage > max_ctree_mem):
            inference = LoopyBPInference(factor_scopes, self, logspace)
        self.inference = inference
    
    def em_learn(self, data_frame, weight_column=None):
        observed = set()
        latent = set()
        for v in self.G.nodes():
            assert v in self.var_names
            if v in data_frame.columns:
                observed.add(v)
            else:
                latent.add(v)
    
        for v in latent:
            'Assign random latent variables to observations' 
            vidx = self.var_index(v)
            card = self.cardinalities[vidx]
            data_frame[v] = pandas.Series(np.random.randint(low=0, high=self.card, size=len(data_frame)))
        self.init_inference()
        
            
            
            
                
            
            
        
        
        