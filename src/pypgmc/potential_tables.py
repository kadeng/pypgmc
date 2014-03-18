
'''

@author: Kai Londenberg

@TODO: Implement an efficient Noisy-Max PotentialTable with sparse parametrization
    as described in http://www.ia.uned.es/~seve/publications/MAX.pdf
'''

__all__ = ['PotentialTable']

import theano
import theano.tensor as T
from discrete_pgm import DiscretePGM
import numpy as np
from expression_utils import LogSumExp

class PotentialTable(object):

    ''' A (not neccessarily normalized) potential table. It may represent a conditional probability table,
    evidence entered into a DiscretePGM, or just an unnormalized mutual potential'''
    def __init__(self, var_set, pt_tensor=None, discrete_pgm=None, name=None):

        if (discrete_pgm is None):
            discrete_pgm = DiscretePGM.get_context()
        if (discrete_pgm is None):
            raise Exception("No DiscretePGM specified, neither explicit nor as current context")
        self.discrete_pgm = discrete_pgm
        self.var_set = frozenset(var_set)
        self.scope = discrete_pgm.map_var_set(self.var_set)

        var_indices = sorted([ discrete_pgm.var_index(v) for v in var_set])
        var_idx_map = {}
        for i, sorted_idx in enumerate(var_indices):
            var_idx_map[sorted_idx] = i

        self.var_indices = var_indices
        self.var_idx_map = var_idx_map

        shp = [self.discrete_pgm.cardinalities[vidx] for vidx in var_indices]
        self.shape = shp
        self.is_shared = False
        if (pt_tensor=="ones"):
            pt_tensor = T.ones(shp, dtype=theano.config.floatX)
        if (pt_tensor=='zeros'):
            pt_tensor = T.zeros(shp, dtype=theano.config.floatX)
        if (pt_tensor=="shared"):
            pt_tensor = theano.shared(np.zeros(shp, dtype=theano.config.floatX))
            self.is_shared = True
        if (pt_tensor is None):
            bcast = [False,]*len(self.var_set)
            tensor_type = T.TensorType(dtype=theano.config.floatX, broadcastable=bcast)
            self.pt_tensor = T.TensorVariable(type=tensor_type, name=name)
        else:
            self.pt_tensor = T.as_tensor_variable(pt_tensor)

    def _ct(self, other):
        ''' Helper function to make tensors dimensions compatible'''
        if (other.var_set == self.var_set):
            return (self.pt_tensor, other.pt_tensor)
        union_var_set = other.scope.union(self.scope)
        vidx1 = frozenset(self.var_indices)
        vidx2 = frozenset(other.var_indices)
        union_indices = vidx1.union(vidx2)

        shape1 = []
        shape2 = []
        b1 = []
        b2 = []
        u1 = []
        u2 = []

        for i,vidx in enumerate(sorted(union_indices)):
            if (vidx in vidx1):
                shape1.append(self.discrete_pgm.cardinalities[vidx])
                u1.append(i)
            else:
                shape1.append(1)
                b1.append(i)
            if (vidx in vidx2):
                shape2.append(self.discrete_pgm.cardinalities[vidx])
                u2.append(i)
            else:
                shape2.append(1)
                b2.append(i)
        t1 = T.addbroadcast(T.unbroadcast(self.pt_tensor.reshape(shape1, len(shape1)), *u1), *b1)
        t2 = T.addbroadcast(T.unbroadcast(other.pt_tensor.reshape(shape2, len(shape2)), *u2), *b2)
        return (t1, t2)


    def _op_result_cpt(self, other, result):
        ''' Helper function - return the result of some operation'''
        assert other.discrete_pgm == self.discrete_pgm
        return PotentialTable(other.scope.union(self.scope), result, self.discrete_pgm)


    def __mul__(self, other):
        ''' Factor Potential Multiplication '''
        t1,t2 = self._ct(other)
        return self._op_result_cpt(other, t1*t2)

    def __add__(self, other):
        ''' Factor Potential Addition
         '''
        t1,t2 = self._ct(other)
        return self._op_result_cpt(other, t1+t2)

    def marginalize(self, remove_var_set=[]):
        """ Marginalize over a given set of variables
            remove_var_set(iterable(str)): Set of variables to marginalize out. May be any iterable (list, tuple, set, whatever)
        Returns:
            A new PotentialTable with the given variables marginalized out
        """
        remove_var_set = self.discrete_pgm.map_var_set(set(remove_var_set))
        remove = self.scope.intersection(remove_var_set)
        keep_vars = self.scope - remove_var_set
        sum_axes = sorted([self.var_idx_map[i] for i in remove])
        res = T.sum(self.pt_tensor, axis=sum_axes, keepdims=False)
        return PotentialTable(keep_vars, res, self.discrete_pgm)

    def logsumexp_marginalize(self, remove_var_set=[]):
        """ Marginalize over a given set of variables in log space using a log(sum(exp(...))) numerical stability optimization
            remove_var_set(iterable(str)): Set of variables to marginalize out. May be any iterable (list, tuple, set, whatever)
        Returns:
            A new PotentialTable with the given variables marginalized out
        """
        remove_var_set = self.discrete_pgm.map_var_set(set(remove_var_set))
        remove = self.scope.intersection(remove_var_set)
        keep_vars = self.scope - remove_var_set
        sum_axes = sorted([self.var_idx_map[i] for i in remove])
        res = LogSumExp(self.pt_tensor, axis=sum_axes, keepdims=False)
        return PotentialTable(keep_vars, res, self.discrete_pgm)

    def max_marginalize(self, remove_var_set=[]):
        """ max-marginalize over a given set of variables
        Args:
            remove_var_set(iterable(str)): Set of variables to max-marginalize out. May be any iterable (list, tuple, set, whatever)

        Returns:
            A new PotentialTable with the given variables max-marginalized out
        """
        remove_var_set = self.discrete_pgm.map_var_set(set(remove_var_set))
        remove = self.scope.intersection(remove_var_set)
        keep_vars = self.scope - remove_var_set
        sum_axes = sorted([self.var_idx_map[i] for i in remove])
        res = T.max(self.pt_tensor, axis=sum_axes, keepdims=False)
        return PotentialTable(keep_vars, res, self.discrete_pgm)

    def min_marginalize(self, remove_var_set=[]):
        """ min-marginalize over a given set of variables
        Args:
            remove_var_set(iterable(str)): Set of variables to min-marginalize out. May be any iterable (list, tuple, set, whatever)

        Returns:
            A new PotentialTable with the given variables min-marginalized out
        """
        remove_var_set = self.discrete_pgm.map_var_set(set(remove_var_set))
        remove = self.scope.intersection(remove_var_set)
        keep_vars = self.scope - remove_var_set
        sum_axes = sorted([self.var_idx_map[i] for i in remove])
        res = T.min(self.pt_tensor, axis=sum_axes, keepdims=False)
        return PotentialTable(keep_vars, res, self.discrete_pgm)

    def normalize(self, target_var, inplace=True):
        """ Normalize this PotentialTable into a conditional distribution of the given target variable.
        Args:
            target_var(str): Target variable
            inplace(Boolean): Whether this operation should be performed inplace, i.e. without returning a new PotentialTable (default: True)

        Returns:
            if inplace=False, return a new normalized PotentialTable. Otherwise, return nothing
            """

        #vset = set(self.var_set)
        #vset.remove(target_var)
        #sum_axe_names = list(vset)
        #sum_axes = [self.discrete_pgm.var_index(v) for v in sum_axe_names]
        normalizer = T.sum(self.pt_tensor, axis=self.var_idx_map[self.discrete_pgm.var_index(target_var)], keepdims=True)
        normalizer = T.switch(T.neq(normalizer, 0.0),normalizer, T.ones_like(normalizer))
        return self._modification_result(self.pt_tensor / normalizer, inplace)

    def to_logspace(self, inplace=True):
        return self._modification_result(T.log(self.pt_tensor), inplace)

    def get_subtensor(self, assignment_dict):
        ''' Get a subtensor of this potential table which agrees with the given assignment of values to variables
        '''
        vset = frozenset(assignment_dict.keys())
        indices = [ slice(None,None), ] * len(self.var_set)
        var_indices = [self.pt_tensor]

        for k in sorted(assignment_dict.keys()):
            v = assignment_dict[k]
            mi = self.var_idx_map.get(self.discrete_pgm.var_index(k))
            if (mi is None):
                continue # This variable doesn't exist in this potential
            indices[mi] = v
            if (isinstance(v, T.TensorVariable)):
                var_indices.append(v)
        pt_slice = T.Subtensor(indices)(*var_indices)
        return pt_slice

    def _modification_result(self, new_pt_tensor, inplace):
        ''' Helper function: returns the result of a modification, which may be either inplace (modifying this
        Potential Table) or not'''
        if (not inplace):
            return PotentialTable(self.var_set, new_pt_tensor, self.discrete_pgm)
        self.pt_tensor = new_pt_tensor
        return self

    def get_value_of_assignment(self, assignment_dict):
        ''' Get the value associated with an assignment of values to variables'''
        vset = frozenset(assignment_dict.keys())
        assert vset == self.var_set
        return self.get_subtensor(assignment_dict)

    def set_value_of_assignment(self, assignment_dict, value, inplace=True):
        ''' Set the value associated with an assignment of values to variables'''
        assert self.scope == self.discrete_pgm.map_var_set(assignment_dict.keys())
        tres = T.set_subtensor(self.get_value_of_assignment(assignment_dict), value)
        return self._modification_result(tres, inplace)

    def set_conditional_prob(self, target_var, parent_assignment_dict, prob_vec, inplace=True):
        ''' Assign a discrete probability distribution to a specific assignment of parent variables

        P(Target=x_i| Pa(Target)=parent_assignment)) = prob_vec[i]

        This operation works inplace, i.e. it modifies this PotentialTable, and does not return anything.
        Args:
            parent_assignment_dict: Assignment to the parent variables, given as a dict (name->int)
            prob_vec: A vector with the length of the remaining target variables' cardinality.

        '''
        tidx = self.discrete_pgm.var_index(target_var)
        tset = frozenset([tidx])
        vset = self.discrete_pgm.map_var_set(frozenset(parent_assignment_dict.keys()))

        if (not (tidx in self.scope)):
            raise Exception("Target Variable not in own variable set")
        if ((self.scope-vset)!=tset):
            raise Exception("Missing Parent Variable in Conditional Probability Assignment")
        indices = [0,]*len(self.var_set)

        for k,v in parent_assignment_dict.iteritems():
            indices[self.var_idx_map[self.discrete_pgm.var_index(k)]] = v
        indices[self.var_idx_map[self.discrete_pgm.var_index(target_var)]] = slice(None,None)
        pt_slice = T.Subtensor(indices)(self.pt_tensor)
        val = T.as_tensor_variable(prob_vec)
        tres = T.set_subtensor(pt_slice, val)
        return self._modification_result(tres, inplace)

    def observe_evidence(self, evidence_dict, inplace=False):
        ''' Returns PotentialTable where all entries which do not agree with the evidence are set to zero.

        Note: Do not use this to enter actual evidence into clique trees, since this would be
        very inefficient. It's better to integrate complete evidence factors into the clique tree.

        Args:
            evidence_dict(dict): A dict which assigns a set of variables to their observed values

        Returns:
            A new PotentialTable, which incorporates the given evidence. I.e. all entries which do not agree with
            the evidence have been set to zero. The resulting PotentialTable will not be normalized.
        '''



        vset = self.discrete_pgm.map_var_set(frozenset(evidence_dict.keys()))
        # Without any evidence, don't limit selection
        indices = [slice(None,None),]*len(self.var_set)
        var_indices = [self.pt_tensor] # The first argument needs to be the tensor that we take the subtensor of
        result_indices = [0] # The first argument needs to be the tensor that we take the subtensor of.
        changes_anything = False
        for k in evidence_dict.keys():
            v = evidence_dict[k]

            vk = self.var_idx_map.get(self.discrete_pgm.var_index(k))
            if (vk is None):
                continue
            changes_anything = True
            indices[vk] = v
            if (isinstance(v, T.TensorVariable)):
                var_indices.append(v)
                result_indices.append(v)
        if (not changes_anything):
            return self._modification_result(self.pt_tensor, inplace)
        result_cpt = T.zeros_like(self.pt_tensor)
        result_indices[0] = result_cpt # The first argument needs to be the tensor that we take the subtensor of
        pt_slice = T.Subtensor(indices)(*var_indices)
        result_slice = T.Subtensor(indices)(*result_indices)
        tres = T.set_subtensor(result_slice, pt_slice, False)
        return self._modification_result(tres, inplace)


def compact_shape(ndarr, var_set, discrete_pgm=None):
        ''' Helper function: Create ndarray in compact format from ndarray in expanded format'''
        if (discrete_pgm is None):
            discrete_pgm = DiscretePGM.get_context()
        if (discrete_pgm is None):
            raise Exception("No DiscretePGM specified, neither explicit nor as current context")
        scope = discrete_pgm.map_var_set(var_set)
        return np.reshape(ndarr, [discrete_pgm.cardinalities[i] for i in sorted(scope)])

def expand_shape(ndarr, var_set, discrete_pgm=None):
        ''' Helper function: Create ndarray in compact format from ndarray in expanded format'''
        if (discrete_pgm is None):
            discrete_pgm = DiscretePGM.get_context()
        if (discrete_pgm is None):
            raise Exception("No DiscretePGM specified, neither explicit nor as current context")
        scope = discrete_pgm.map_var_set(var_set)
        shp = [1,]*len(discrete_pgm.cardinalities)
        for i in scope:
            shp[i] = discrete_pgm.cardinalities[i]
        return np.reshape(ndarr, shp)
