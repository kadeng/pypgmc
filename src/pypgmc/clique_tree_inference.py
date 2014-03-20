'''

CliqueTreeInference class, which implements pypgmc clique tree inference based on
elementary operations provided by the PotentialTable class, working in the context
of a DiscretePGM model.


@author: Kai Londenberg (Kai.Londenberg@googlemail.com)

'''
from discrete_pgm import *
from potential_tables import *
import numpy as np
import theano
import theano.tensor as T

from expression_utils import add_op, mul_op

class CliqueTreeInference(object):


    def __init__(self, factor_scopes=[], discrete_pgm=None, logspace=False):
        """ Construct CliqueTreeInference object from set of factor scopes
        Args:
            factor_scopes(list(scopes)): List of variable scopes (given by name or variable index) or PotentialTable scopes.
                                        in the latter case, scopes from these PotentialTables will be taken.
            discrete_pgm(DiscretePGM): DiscretePGM model we are working in. May be null if we are "with" in that model's context

            logspace: Whether to operate in log-space (in these case factor are added instead of multiplies, and we need to do logsumexp operations instead
                     of simple summing when marginalizing

        Returns:
            A new CliqueTreeInference object configured to work with these
        """
        if (discrete_pgm is None):
            discrete_pgm = DiscretePGM.get_context()
        if (discrete_pgm is None):
            raise Exception("No DiscretePGM specified, neither explicit nor as current context")
        self.discrete_pgm = discrete_pgm
        self.logspace = logspace
        if (logspace):
            self.pop = 'add'
        else:
            self.pop = 'multiply'

        if (not logspace):
            self.op = mul_op
            self.neutral = 'ones'
        else:
            self.op = add_op
            self.neutral = 'zeros'

        fscopes = []
        for f in factor_scopes:
            if (isinstance(f, PotentialTable)):
                fscopes.append(f.scope)
            else:
                fscopes.append(self.discrete_pgm.map_var_set(f))
        factor_scopes = set(fscopes)
        for fac in fscopes:
            contained = False
            for es in fscopes:
                if fac.issubset(es):
                    contained = True
                    break
                if es.issubset(fac):
                    if (es in factor_scopes):
                        factor_scopes.remove(es)
            if (not contained):
                factor_scopes.add(fac)

        self.factor_scopes = frozenset(factor_scopes)
        self._create_clique_tree()


    def _create_clique_tree(self):
        ''' Create the junction tree, i.e. determine the cliques and their messaging scheme'''

        factor_scopes = set(self.factor_scopes)
        ' Set of frozensets containing the scopes (frozenset of  var indices) of all factors - this set is being modified a lot during this algorithm'

        var_edges = np.zeros((self.discrete_pgm.numvars,self.discrete_pgm.numvars), dtype=np.int8)
        'Adjacency Matrix of the variables'

        clique_scopes = []
        'List of frozensets containing variable scopes (frozenset of var indices) of the cliques'

        eliminated_vars = []
        'Variable elimination ordering that has been used'

        clique_connected = [False]*self.discrete_pgm.numvars
        'Boolean array, indicating whether a clique has been connected already'

        clique_edges = np.zeros((self.discrete_pgm.numvars,self.discrete_pgm.numvars), dtype=np.int8)
        'Adjacency matrix of the cliques - we make this matrix as large as possibly needed'

        for scp in factor_scopes:
            for v1 in scp:
                for v2 in scp:
                    var_edges[v1,v2] = 1

        def eliminate_var(elim_var_idx, var_edges, clique_connected, factor_scopes, clique_scopes, eliminated_vars, clique_edges):
            ''' Eliminate a given variable from the set of factor scopes, create a new clique with all factors sharing
                that variable, update the set of factor scopes and update the clique adjacency matrix
                '''

            used_factors = set()
            new_clique_scope = set()
            for v in factor_scopes:
                if (elim_var_idx in v):
                    used_factors.add(v)
                    new_clique_scope = new_clique_scope | v
            used_factors = frozenset(used_factors)
            new_clique_scope = frozenset(new_clique_scope)
            unused_factors = factor_scopes - used_factors

            eliminated_var_set = frozenset([elim_var_idx])

            # Remove all factors which we have used
            factor_scopes = factor_scopes - used_factors

            # Create a new factor with the scope of this clique minus the eliminated variable
            # and add it to the list of factor scopes
            new_factor_scope = new_clique_scope - eliminated_var_set
            factor_scopes.add(new_factor_scope)

            new_clique_idx = len(clique_scopes)
            clique_scopes.append(new_clique_scope)
            eliminated_vars.append(elim_var_idx)
            for i in range(new_clique_idx):
                if ((not clique_connected[i]) and (clique_scopes[i] & eliminated_var_set)):
                    clique_edges[i,new_clique_idx] = 1
                    clique_edges[new_clique_idx, i] = 1
                    clique_connected[i] = True # Mark that clique as connected

            var_edges[elim_var_idx,:] = 0
            var_edges[:,elim_var_idx] = 0
            for v1 in new_factor_scope:
                    for v2 in new_factor_scope:
                        var_edges[v1,v2] = 1
            return (clique_connected, factor_scopes, clique_scopes, eliminated_vars, clique_edges, var_edges)

        def prune_tree(clique_scopes, clique_edges):
            ''' Eliminate unneccessary cliques by joining them with neighbours if possible'''
            removed_clique_indices=set()
            while True:
                modified = False
                for i1 in range(len(clique_scopes)):
                    if (i1 in removed_clique_indices):
                        continue
                    c1 = clique_scopes[i1]

                    for i2 in  range(len(clique_scopes)):
                        if (i1==i2):
                            continue
                        if (i2 in removed_clique_indices):
                            continue
                        if (clique_edges[i1,i2]):
                            c2 = clique_scopes[i2]
                            if (c2.issuperset(c1)):
                                removed_clique_indices.add(i1)
                                clique_edges[i2,:] |= clique_edges[i1,:]
                                clique_edges[:,i2] |= clique_edges[:,i1]
                                clique_edges[i2,i2] = 0
                                modified = True
                                break
                    if (modified):
                        break
                if (not modified):
                    break
            remaining_clique_indices = sorted(list(set(range(len(clique_scopes)))-set(removed_clique_indices)))
            clique_scopes = [clique_scopes[i] for i in remaining_clique_indices]
            clique_edges = clique_edges[remaining_clique_indices,:][:,remaining_clique_indices]
            return (clique_scopes, clique_edges)

        # Eliminate all variables, creating cliques on your way
        numvars = self.discrete_pgm.numvars
        for cc in range(numvars):
            best_elimination_var = -1;
            best_score = 99999999999
            scores = var_edges.sum(axis=1)
            for i in range(numvars):
                score = scores[i]
                if (score > 0 and score < best_score):
                      best_score = score
                      best_elimination_var = i
            if (best_elimination_var!=-1):
                clique_connected, factor_scopes, clique_scopes, eliminated_vars, clique_edges, var_edges = eliminate_var(best_elimination_var, var_edges, clique_connected, factor_scopes, clique_scopes, eliminated_vars, clique_edges)

        # Prune the tree of unneccessary cliques
        clique_scopes, clique_edges = prune_tree(clique_scopes, clique_edges)

        # These are our results
        self.clique_connectivity = np.sum(clique_edges, axis=1)
        self.leaf_clique_indices =np.nonzero(self.clique_connectivity==1)[0] # Find all cliques which are connected to just one other clique

        self.clique_scopes = clique_scopes
        self.clique_edges = clique_edges
        self.elimination_order = eliminated_vars

    def get_mem_usage(self, float_size=8):
        ''' Estimate memory usage when calibrating an entire tree.
            This simply calculates the combined size of all the clique tensors and the messaging tensors.
            An important size to know, before you actually try to allocate the memory.

            It might actually be even much better to use theano.tensor.utils.shape_of_variables(fgraph, input_shapes)
            see http://deeplearning.net/software/theano/library/tensor/utils.html
            as this will probably give a much more accurate number for a given query
            '''
        sumsize = long(0)
        for i in range(len(self.clique_scopes)):
            csize = long(float_size)
            for v in self.clique_scopes[i]:
                csize *= long(self.discrete_pgm.cardinalities[v])
            sumsize += long(csize)
        for i in range(len(self.clique_scopes)):
            # Calculate size of clique potential
            csize = long(float_size)
            for v in self.clique_scopes[i]:
                csize *= long(self.discrete_pgm.cardinalities[v])
            sumsize += csize
            # Calculate size for each message potential
            for i2 in range(len(self.clique_scopes)):
                if (self.clique_edges[i,i2]):
                    message_scope = self.clique_scopes[i] & self.clique_scopes[i2]
                    msize = long(float_size)
                    for v in message_scope:
                        msize *= long(self.discrete_pgm.cardinalities[v])
                    sumsize += long(msize)
        return sumsize

    def _create_initial_potentials(self, factors=[]):
        '''
        Assign factors to cliques, and calculate initial potentials by multiplying these initial factors
        raises an Exception if one of the factors does not fit into any clique
        '''
        clique_map = [None]*len(self.clique_scopes) #
        clique_init = [-1]*len(self.clique_scopes)
        fs = [f.scope for f in factors]
        for fi in range(len(factors)):
            assigned = False
            for ci in range(len(self.clique_scopes)):
                if (fs[fi].issubset(self.clique_scopes[ci])):
                    assigned = True
                    if (clique_map[ci] is None):
                        clique_map[ci] = []
                    clique_map[ci].append(fi)
                    if (fs[fi] == self.clique_scopes[ci]):
                        clique_init[ci] = fi  # We have a matching assigned factor, so don't bother initializing this with ones
                    break
            if (not assigned):
                raise Exception("Factor %d does not fit into the clique tree. Scope (%r) " % (fi, factors[fi].var_set))

        clique_potentials = []
        for ci in range(len(self.clique_scopes)):
            if (clique_init[ci]==-1):
                cpot = PotentialTable(self.clique_scopes[ci], self.neutral)
            else:
                cpot = factors[clique_init[ci]]
            for fi in clique_map[ci]:
                if (fi==clique_init[ci]):
                    continue
                cpot = self.op(cpot, factors[fi])
            clique_potentials.append(cpot)
        return clique_potentials

    def _create_initial_message_potentials(self):
        'Trivial helper function'
        messages = [None]*len(self.clique_scopes)
        for c1 in range(len(self.clique_scopes)):
            messages[c1] = [None]*len(self.clique_scopes)
        return messages

    def _calc_message_order(self):
        ''' Given the clique list and their adjacency matrix, calculate a message passing order. '''
        unmessaged = np.copy(self.clique_edges)
        message_ordering = list()
        reverse_messages = list()
        while True:
            next_cliques = np.nonzero(np.sum(unmessaged, axis=1)==1)[0] # Find all cliques which can pass a message immediately
            if (len(next_cliques)==0):
                break
            for src_clique in next_cliques:
                target_cliques = np.nonzero(unmessaged[:,src_clique])[0] # Find index of the one unmessaged neighbour
                if (len(target_cliques)==0):
                    break # This happens when this is the last clique remaining
                target_clique = target_cliques[0]
                message_ordering.append((src_clique, target_clique))
                reverse_messages.append((target_clique, src_clique))

                unmessaged[src_clique,target_clique]=0
                unmessaged[target_clique, src_clique]=0
        assert np.all(unmessaged==0)
        res = list(message_ordering) + list(reversed(reverse_messages))
        return res

    def calibrated_potentials(self, factors, algo='sum_product'):
        ''' Given a list of PotentialTable factors, return a list of
            PotentialTables containing symbolic tensor representations of calibrated clique potentials.
            These potentials agree on marginals and can be used for most relevant queries.

            Args:
            factors(list(PotentialTable)): List of potential table factors to include.
                                           These may be normalized conditional probability tables or evidence factors

            algo: May be either 'sum_product', 'max_product' or 'min_product'


        Returns:
            list of potential tables containing symbolic theano tensors representing the calibrated cliques
            '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))
        messages = [None]*len(self.clique_scopes)
        for c1 in range(len(self.clique_scopes)):
            messages[c1] = [None]*len(self.clique_scopes)
        initial_potentials = self._create_initial_potentials(factors)
        message_order = self._calc_message_order()

        messaged = np.zeros_like(self.clique_edges)
        for src, target in message_order:
            potential = initial_potentials[src]
            nz = np.nonzero(messaged[:,src])[0]
            relevant_messages = set([int(z) for z in nz])-set([target])
            for mi in relevant_messages:
                potential = self.op(potential, messages[mi][src])
            message_scope = self.clique_scopes[src] & self.clique_scopes[target]

            if (algo=='sum_product'):
                if (self.logspace):
                    messages[src][target] = potential.logsumexp_marginalize(potential.scope - message_scope)
                else:
                    messages[src][target] = potential.marginalize(potential.scope - message_scope)
            if (algo=='max_product'):
                messages[src][target] = potential.max_marginalize(potential.scope - message_scope)
            if (algo=='min_product'):
                messages[src][target] = potential.min_marginalize(potential.scope - message_scope)
            messaged[src,target] = 1
        assert np.all(messaged==self.clique_edges)
        calibrated_potentials = []
        for c in range(len(self.clique_scopes)):
            incoming_messages = np.nonzero(messaged[:,c])[0]

            potential = initial_potentials[c]
            for i in incoming_messages:
                potential = self.op(potential, messages[i][c])
            calibrated_potentials.append(potential)
        return calibrated_potentials

    def probability(self, factors):
        '''Return a symbolic theano expression for the joint marginal probability of all given factors
           Args:
               factors(list(PotentialTable)): List of potential table factors to include.
                                           These may be normalized conditional probability tables or evidence factors

           Returns:
               A symbolic theano scalar which depends on the given factors. If the given factors consist of conditional
               probability tables and evidence factors, the result calculates the probability of the evidence.
        '''
        return self._prob_expr(factors, False)

    def logp(self, factors):
        '''Return a symbolic theano expression for the joint marginal probability of all given factors
           Args:
               factors(list(PotentialTable)): List of potential table factors to include.
                                           These may be normalized conditional probability tables or evidence factors

           Returns:
               A symbolic theano scalar which depends on the given factors. If the given factors consist of conditional
               probability tables and evidence factors, the result calculates the probability of the evidence.
        '''
        return self._prob_expr(factors, True)

    def _prob_expr(self, factors, as_logp):
        ''' Implementation for probability and logp functions respectively'''

        # Get Calibrated Potentials
        calibrated = self.calibrated_potentials(factors, 'sum_product')
        minlen = 9999999999
        mint = None
        # Find a clique with minimal scope, so we minimize the amount of final summing
        for v in calibrated:
            if (len(v.scope)<minlen):
                minlen = len(v.scope)
                mint = v
                if (minlen==1):
                    break
        # Marginalize out everything
        if (self.logspace):
            res = mint.logsumexp_marginalize(mint.scope)
            res = T.reshape(res.pt_tensor, [1], ndim=1)[0]
            if (not as_logp):
                res = T.exp(res)
        else:
            res = mint.marginalize(mint.scope)
            res = T.reshape(res.pt_tensor, [1], ndim=1)[0]
            if (as_logp):
                res = T.log(res)
        return res
