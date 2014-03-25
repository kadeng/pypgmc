
'''

MessagePassingGraph class, which implements message passing inference based on
elementary operations provided by the PotentialTable class, working in the context
of a DiscretePGM model.


@author: Kai Londenberg (Kai.Londenberg@googlemail.com)

'''

from discrete_pgm import *
from potential_tables import *
import numpy as np
import theano
import theano.tensor as T
import theano.scan_module
from expression_utils import add_op, mul_op

class MessagePassingGraph(object):
   
    def __init__(self, factor_scopes=[], discrete_pgm=None, logspace=False):
        """ Construct MessagePassingGraph object from set of factor scopes.
            This is intended as a common base class for a clique tree and a loopy bp inference implementation.
            
        Args:
            factor_scopes(list(scopes)): List of variable scopes (given by name or variable index) or PotentialTable scopes.
                                        in the latter case, scopes from these PotentialTables will be taken.
            discrete_pgm(DiscretePGM): DiscretePGM model we are working in. May be null if we are "with" in that model's context

            logspace: Whether to operate in log-space (in these case factor are added instead of multiplies, and we need to do logsumexp operations instead
                     of simple summing when marginalizing
            graph_construcor: Graph construction method, which can either create a clique tree or a loopy graph. Usually you will pass either
                    the create_loopy_graph function, or the create_clique_tree_graph function. 

        Returns:
            A new MessagePassingGraph object
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
        self._create_clique_graph()

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

    def create_mp_state(self, factors, algo='sum_product'):
        ''' Create an _MPState object which wraps the state of possibly multiple messaging passes
            for the given algorithm and list of factors'''
        return _MPState(self, factors, algo)

    
    def _calc_message_order(self):
        ''' 
        Given the clique list and their adjacency matrix, calculate a message passing order. 
        Constructs a message passing order which either completely calibrates the clique graph (if it is a tree),
        or, if it is not, is suited to be called iteratively. In any way, the message order will include all messages,
        i.e. cover all possible edges in the clique graph. It is not neccessarily an optimal order if the graph is not a tree.
        
        For loopy graphs, repeated calls of this method might return a random alternative messaging scheme.
        In order to get reproducible results, call np.random.seed(..) before a call to this method. 
        '''
        unmessaged = np.copy(self.clique_edges)
        message_ordering = list()
        reverse_messages = list()
        while True:
            next_cliques = np.nonzero(np.sum(unmessaged, axis=1) == 1)[0]  # Find all cliques which can pass a message immediately
            if (len(next_cliques) == 0):
                if (np.all(unmessaged == 0)):
                    break
                else:
                    # Special code for loopy graphs:
                    # Find a random clique with a minimal positive number of incoming messages
                    nsum = np.sum(unmessaged, axis=1)
                    nnz = nsum[nsum>0]
                    if (len(nnz)==0):
                        break
                    ngmval = np.min(nnz)
                    next_cliques = np.nonzero(nsum == ngmval)[0]
                    if (len(next_cliques) == 0):
                        break
                    next_cliques = [next_cliques[np.random.randint(0, len(next_cliques))]]
            if (len(next_cliques) == 0):
                break
            for src_clique in next_cliques:
                target_cliques = np.nonzero(unmessaged[:, src_clique])[0]  # Find index of the one unmessaged neighbour
                if (len(target_cliques) == 0):
                    break  # This happens when this is the last clique remaining
                target_clique = target_cliques[0]
                message_ordering.append((src_clique, target_clique))
                reverse_messages.append((target_clique, src_clique))

                unmessaged[src_clique, target_clique] = 0
                unmessaged[target_clique, src_clique] = 0
        assert np.all(unmessaged == 0)
        res = list(message_ordering) + list(reversed(reverse_messages))
        return res    
    
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
            a theano scalar value of 0 indicating that this always converges in one step.
            '''
        raise NotImplementedError()
    
    def _create_clique_graph(self):
        ''' 
        Called during initialization. Combine factors into larger cliques
        using an appropriate algorithm. 
        
        Populates the properties:
            self.clique_scopes: list of clique scopes in the form of frozenset(int) values.
            self.clique_edges: 2-Dimensional ndarray with an entry of 1 for all cliques sharing at least one variable. This gives the graph connectivity
        
        See CliqueTreeInference.create_clique_graph() and 
        LoopyBPInference.create_clique_graph() for reference implementations.
        '''
        raise NotImplementedError()

class CliqueTreeInference(MessagePassingGraph):
    ''' The clique tree inference is an efficient exact inference algorithm for discrete
        PGMs. It's the preferred algorithm to use if the resulting data structures are not prohibitevely large.'''
    
    def __init__(self, factor_scopes=[], discrete_pgm=None, logspace=False):
        """ Construct CliqueTreeInference object from set of factor scopes.
            
        Args:
            factor_scopes(list(scopes)): List of variable scopes (given by name or variable index) or PotentialTable scopes.
                                        in the latter case, scopes from these PotentialTables will be taken.
            discrete_pgm(DiscretePGM): DiscretePGM model we are working in. May be null if we are "with" in that model's context

            logspace: Whether to operate in log-space (in these case factor are added instead of multiplies, and we need to do logsumexp operations instead
                     of simple summing when marginalizing
            graph_construcor: Graph construction method, which can either create a clique tree or a loopy graph. Usually you will pass either
                    the create_loopy_graph function, or the create_clique_tree_graph function. 

        Returns:
            A new CliqueTreeInference object
        """
        super(CliqueTreeInference, self).__init__(factor_scopes, discrete_pgm, logspace)
           
    def _create_clique_graph(self):
        ''' Create the junction tree, i.e. determine the cliques and their messaging scheme. 
        Function that is intended to be passed to the MessagePassingGraph constructor and used as a method'''

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
            a theano scalar value of 0 indicating that this always converges in one step.
            '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))
        mpstate = self.create_mp_state(factors, algo)
        (message_order, updated_messages) = mpstate.pass_messages()
        calibrated_clique_potentials = mpstate.get_calibrated_potentials(updated_messages)
        return calibrated_clique_potentials
    
    
class LoopyBPInference(MessagePassingGraph):
    ''' Loopy Belief Propagation Inference implementation based on theano.
    
    Loopy Belief Propagation is a scalable approximate inference algorithm which is capable to solve problems which are
    intractable for CliqueTrees due to memory and time constraints. If the solution is tractable using CliqueTreeInference, you should
    prefer that. If not, try this one. You can check the plausibility of using CliqueTreeInference by calculating the memory usage
    using the get_mem_usage() method, which is available for both CliqueTree Inference and LoopyBPInference.
    '''
    
    def __init__(self, factor_scopes=[], discrete_pgm=None, logspace=False, max_iters=10, convergence_threshold=-1.0):
        """ Construct LoopyBPInference object from set of factor scopes.
            
        Args:
            factor_scopes(list(scopes)): List of variable scopes (given by name or variable index) or PotentialTable scopes.
                                        in the latter case, scopes from these PotentialTables will be taken.
            discrete_pgm(DiscretePGM): DiscretePGM model we are working in. May be null if we are "with" in that model's context

            logspace: Whether to operate in log-space (in these case factor are added instead of multiplies, and we need to do logsumexp operations instead
                     of simple summing when marginalizing
            max_iters: Maximum number of iterative steps - defaults to 10'''
            convergence_threshold: How much the message tensors may change in sum before the iteration stops. 
                                If negative, we run a fixed number of iterations. Default: Negative (-1.0)
            
        Returns:
            A new LoopyBPInference object
        """
        super(LoopyBPInference, self).__init__(factor_scopes, discrete_pgm, logspace)
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        
    def _create_clique_graph(self):
        ''' Create a potentially loopy message passing graph
        Create the junction tree, i.e. determine the cliques and their messaging scheme. 
        Function that is intended to be passed to the MessagePassingGraph constructor and used as a method
        '''

        factor_scopes = set(self.factor_scopes)
        ' Set of frozensets containing the scopes (frozenset of  var indices) of all factors - this set is being modified a lot during this algorithm'

        var_edges = np.zeros((self.discrete_pgm.numvars,self.discrete_pgm.numvars), dtype=np.int8)
        'Adjacency Matrix of the variables'

        clique_scopes = [frozenset(v) for v in factor_scopes]
        'List of frozensets containing variable scopes (frozenset of var indices) of the cliques'

        clique_edges = np.zeros((self.discrete_pgm.numvars,self.discrete_pgm.numvars), dtype=np.int8)
        'Adjacency matrix of the cliques - we make this matrix as large as possibly needed'

        for c1 in range(len(clique_scopes)):
            for c2 in range(len(clique_scopes)):
                if (c1==c2):
                    continue
                if (clique_scopes[c1] & clique_scopes[c2]):
                    clique_edges[c1, c2] = 1

        for scp in factor_scopes:
            for v1 in scp:
                for v2 in scp:
                    var_edges[v1,v2] = 1

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

        # Prune the tree of unneccessary cliques
        clique_scopes, clique_edges = prune_tree(clique_scopes, clique_edges)

        # These are our results
        self.clique_connectivity = np.sum(clique_edges, axis=1)
        self.leaf_clique_indices =np.nonzero(self.clique_connectivity==1)[0] # Find all cliques which are connected to just one other clique

        self.clique_scopes = clique_scopes
        self.clique_edges = clique_edges
        self.initialized = False
        
    def calibrated_potentials(self, factors, algo='sum_product', return_convergence_info=False):
        ''' Given a list of PotentialTable factors, return a list of
            PotentialTables containing symbolic tensor representations of calibrated clique potentials.
            These potentials agree on marginals and can be used for most relevant queries.
            
            WARNING: Setting a non-negative convergence threshold (i.e. enabling the corresponding code) has a huge performance impact !
            It's definitely recommended to run with a fixed number of iterations, even if this means a large number.

            Args:
            factors(list(PotentialTable)): List of potential table factors to include.
                                           These may be normalized conditional probability tables or evidence factors

            algo: May be either 'sum_product', 'max_product' or 'min_product'
            
            return_convergence_info: Whether to return the convergence info in order to evaluate whether the iteration converged or not

        Returns a tuple consisting of:
            * list of potential tables containing symbolic theano tensors representing the calibrated cliques
            * if return_convergence_criterion is True: Expression for convergence criterion and the number of iterations
            '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))
        
        convergence_threshold=self.convergence_threshold
        max_steps = self.max_iters 
        
        'Create an initial set of messages'
        mpstate = self.create_mp_state(factors, algo)
        (message_order, first_messages) = mpstate.pass_messages()
        
        'Initial value for the convergence checking criterion '
        convergence_criterion = T.ones((1), dtype=theano.config.floatX)[0]
        initial_values =  [convergence_criterion]+ [first_messages[midx].pt_tensor for midx in message_order]
        'We are going to iterate via a pretty complex theano scan op.'
        
        def pass_fn(*inputs):
            ''' 
            Function for scan op. Has to work with variable number of arguments. 
            Input layout: diff, message[message_order[0]], ..., message[message_order[N], initial_potential[0], ..., initial_potential[M] 
            '''
            
            input_messages = {}
            ''' Quick creation of message potential tables by using a shallow copy of existing potential tables'''
            for i,midx in enumerate(message_order):
                input_messages[midx] = first_messages[midx].replace_tensor(inputs[i+1])
            
            off = 1+len(message_order) #  offset into input for the initial potentials
            ipotentials = []
            ''' Create initial potentials from passed inputs'''
            for i, pot in enumerate(mpstate.initial_potentials):
                ipotentials.append(pot.replace_tensor(inputs[off+i]))
            
            ''' Pass messages and calculate next set of messages '''
            (used_message_order, next_messages) = mpstate.pass_messages(input_messages=input_messages, initial_potentials=ipotentials)
            if (convergence_threshold>=0.0): 
                
                ''' Calculate absolute difference between last set of differences and current set for convergence diagnostics'''
                diff = T.sum( T.abs_(next_messages[used_message_order[0]].pt_tensor.flatten() - input_messages[used_message_order[0]].pt_tensor.flatten()))
                for i in range(1, len(used_message_order)):
                    diff += T.sum( T.abs_(next_messages[used_message_order[i]].pt_tensor.flatten() - input_messages[used_message_order[i]].pt_tensor.flatten()))
                
                ''' Create result which conforms to the start of the input layout'''
                resvalues = [diff] + [next_messages[midx].pt_tensor for midx in message_order]
                ''' Return updated values plus a convergence criterion'''
                return resvalues, theano.scan_module.until(diff<=convergence_threshold)
            else:
                diff = convergence_criterion
                resvalues = [diff] + [next_messages[midx].pt_tensor for midx in message_order]
                return resvalues
            
            

        
        ' Calculate the theano expressions for the iterative message passing scheme'''
        result, updates = theano.scan(fn=pass_fn,
                                        outputs_info=initial_values,
                                        non_sequences=[p.pt_tensor for p in mpstate.initial_potentials], 
                                        n_steps=max_steps)
        
        ''' The result[0] are the time slices of the *diff' variable. So diff[-1] is the last diff, and len(result[0]) is the number of iteration steps that we did.'''
        
            
        ''' Create final message potential dictionary'''
        result_messages = {}
        for i,midx in enumerate(message_order):
            result_messages[midx] = first_messages[midx].replace_tensor(result[i+1][-1])
            
        ''' Calculate calibrated potentials from these'''
        calibrated_potentials = mpstate.get_calibrated_potentials(result_messages)
        if (not return_convergence_info):
            return calibrated_potentials
        else:
            convergence = result[0][-1]
            niters = result[0].shape[0]
            return (calibrated_potentials, convergence, niters)
        
class _MPState(object):
    ''' Internal class which represents the state of a Message Passing operation given a single set of input factors.
        May be used to construct various message passing and iteration schemes, from closed form exact solutions (CliqueTree)
        to iterative (LoopyBP) ones. 
        
        Works completely symbolically by working with PotentialTable instances.
        
    ''' 
    def __init__(self, mpgraph, factors, algo='sum_product'):
        ''' Construcot of the _MPState object.
        
        Args:
            mpgraph: MessagePassingGraph instance to construct clique graph for.
            factors: Factors, including possible evidence factors, which form the initial potentials
            algo: Algorithm. May be either 'sum_product', 'max_product' or 'min_product'
            message_order: Default messaging order (list of (src, target) tuples to use in pass_messages
        '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))
        self.factors = factors
        self.algo = algo
        self.mpgraph = mpgraph
        self.messages = {}        
        self.initial_potentials = mpgraph._create_initial_potentials(factors)
            
    def pass_messages(self, message_order=None, input_messages=None, initial_potentials=None):
        '''
        Iterate message passing
        Args:
            message_order: List of (src_clique_idx, target_clique_idx) tuples which gives the messages that should be passed
            input_messages: Dict with (src_clique_idx, target_clique_idx) tuple keys, which are merged with the previous messaging state. 
                            May be used to pass in a shared state, for example.
        Returns:
            Tuple consisting of: (message_order, dict of updated clique potentials, dict of message potentials)
            
            The result is also appended to this objects "iteration_results" list.
        '''   
        messages = dict(self.messages)
        if (input_messages is not None):
            for k in input_messages.keys():
                messages[k] = input_messages[k]
        mpgraph = self.mpgraph
        if (initial_potentials is None):
            initial_potentials = self.initial_potentials
        if (message_order is None):
            message_order = mpgraph._calc_message_order()
        algo = self.algo
        messaged = np.zeros_like(mpgraph.clique_edges)
        for src, target in message_order:
            potential = initial_potentials[src]
            nz = np.nonzero(mpgraph.clique_edges[:,src])[0]
            relevant_messages = set([int(z) for z in nz])-set([target])
            for mi in relevant_messages:
                msgidx = (mi,src)
                if (msgidx in messages):
                    potential = mpgraph.op(potential, messages[msgidx])
            message_scope = mpgraph.clique_scopes[src] & mpgraph.clique_scopes[target]

            if (algo=='sum_product'):
                if (mpgraph.logspace):
                    messages[(src,target)] = potential.logsumexp_marginalize(potential.scope - message_scope)
                else:
                    messages[(src, target)] = potential.marginalize(potential.scope - message_scope)
            if (algo=='max_product'):
                messages[(src,target)] = potential.max_marginalize(potential.scope - message_scope)
            if (algo=='min_product'):
                messages[(src,target)] = potential.min_marginalize(potential.scope - message_scope)
            messaged[src,target] = 1

        updated_messages = dict(messages)
        result =  (message_order, updated_messages)
        return result
    
    def get_calibrated_potentials(self, messages, initial_potentials=None):
        ''' Returns calibrated potential expressions given a set of messages'''
        if (initial_potentials is None):
            initial_potentials = self.initial_potentials
        updated_potentials = []
        mpgraph = self.mpgraph
        for c in range(len(mpgraph.clique_scopes)):
            incoming_messages = np.nonzero(mpgraph.clique_edges[:,c])[0]
            potential = initial_potentials[c]
            for i in incoming_messages:
                msgidx = (i,c)
                if (msgidx in messages):
                    potential = mpgraph.op(potential, messages[msgidx])
            updated_potentials.append(potential)
        return updated_potentials

