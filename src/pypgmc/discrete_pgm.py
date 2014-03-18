'''

Created on May 24, 2013

@author: Kai Londenberg ( Kai.Londenberg@googlemail.com )
'''
import numpy as np
from pymc3_adapter import Context

class DiscretePGM(Context):
    '''
    A discrete Probabilistic Graphical Model
    can be a directed, undirected or hybrid directed/undirected probabilistic model.

    This class represents a context, to which Conditional Probability/Potential Tables may be added

    It manages variables, which, in the context of this PGM, are given numeric indices.

    '''

    def __init__(self, cardinalities=[], var_names=[]):
        '''Create a discrete probabilistic graphical model

        Args:
            var_names(list of str): Names of the discrete variables used in this model.
            cardinalities(list of int): The corresponding variable cardinalities - this corresponds to the shape of this distribution
                                were it expanded to a full joint probability mass.
                                If there are un-named variables (referenced by index only),their cardinalities have to be at the end
        '''
        self.var_names = var_names
        self.cardinalities = cardinalities
        self.numvars = len(cardinalities)
        self.var_indices = dict(zip(var_names, range(len(var_names))))


    def add_var(self, cardinality, name=None):
        """Add a named variable to the model
        Args:
            name(str): Unique name of the variable to be added
            cardinality(int): Cardinality (maximum number of unique values) of the variable
        Returns:
            int, Index of the variable that has just been added
        Raises:
            Exception, if the DiscretePGM already contains CPTs
        """
        if (len(self.cpts)>0):
            raise Exception("Cannot add variable after CPTs have been added to the DiscretePGM")
        if (not (name is None)):
            if (len(self.var_names)!=len(self.cardinalities)):
                raise Exception("Cannot add named variables after unnamed ones have been added to the model")
            self.var_names.append(name)
            self.var_indices = dict(zip(self.var_names, range(len(self.var_names))))
        self.cardinalities.append(cardinality)
        self.numvars = len(self.cardinalities)
        return len(self.var_names)-1

    def var_index(self, var_name):
        ''' Map the name of a variable to it's index.  Transparently handles unnamed variables. If you pass in an integer or long, you'll get
            exactly that number back'''
        if (isinstance( var_name, ( int, long ) )):
            assert var_name>=0
            assert var_name<len(self.cardinalities)
            return var_name
        return self.var_indices[var_name]

    def map_var_set(self, var_set):
        '''Utility method to return a frozenset containing the indices of all variables in the given var_set'''
        result = set()
        for v in var_set:
            result.add(self.var_index(v))
        return frozenset(result)




class BeliefNet(DiscretePGM):

    def __init__(self):
        DiscretePGM.__init__(self)


