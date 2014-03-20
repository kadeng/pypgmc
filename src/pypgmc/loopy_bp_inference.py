'''

LoopyBPInference implements  Loopy Belief Propagation approximate inference based on
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

def simple_message_scheduler(loopy, input, threshold=0.001, verbose=True, max_iters=20, algo='sum_product', message_list=None):
    ''' Simple message scheduling algorithm for Loopy Belief Propagation.
        May be replaced using custom code by LoopyBPInference.setMessageScheduler(function)

        This scheduler expects a LoopyBPInference object that's ready to pass messages.
        i.e. the following methods should have been called earlier:
            * loopy.alloc_resources() to allocate shared resources
            * loopy.set_factors(..) to set the factors and create appropriate messaging and reset functions

        Args:
            loopy: Instance of LoopyBPInference to use
            input: Inputs required to create initial potentials, i.e. which are passed to loopy.reset(..)
            threshold: Stop iterating if we don't get more thatn threshold change in summed message changes from one set of iterations to another.
            algo: Must be 'sum_product', 'max_product' or 'min_product'

        '''
    if (message_list is None):
        message_list = loopy.calc_a_message_order()

    sum_changes = threshold + 1.0
    last_sum_changes = sum_changes*2.0
    iter = 0
    from sys import stdout

    def log_none(msg):
        pass

    def log_verbose(msg):
        print msg
        stdout.flush()

    if (verbose):
        log = log_verbose
    else:
        log = log_none

    log("Resetting Loopy Belief Propagation Inference")
    loopy.reset(input)
    while ((last_sum_changes-sum_changes)>threshold):
        if (iter>=max_iters):
            log("Reached maximum number of iterations (%d)" % iter)
            break
        iter += 1
        last_sum_changes = sum_changes
        sum_changes = 0.0
        for (from_idx, to_idx) in message_list:

            res = loopy.pass_message(from_idx, to_idx, algo)
            log("\t\tMessage %d -> %d - result: %r" % (from_idx, to_idx, res))
            sum_changes += res
        log("  Iteration %d: Summed changes: %f, difference: %f " %  (iter, sum_changes, last_sum_changes-sum_changes))

class LoopyBPInference(object):


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
        self._create_graph()


    def _create_graph(self):
        ''' Create the message passing graph'''

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

    def alloc_resources(self):
        ''' Allocate shared resources for computation'''
        self.shared_messages = SharedMessagePotentials(self.clique_scopes, self.discrete_pgm)
        self.initial_potentials = []
        import sys
        for i, scope in enumerate(self.clique_scopes):
            sys.stdout.flush()
            self.initial_potentials.append(PotentialTable(scope, 'shared', self.discrete_pgm, 'IP_%d' % i))
        self.initial_potential_expressions = None
        self.message_expressions =  { 'max_product' : None, 'sum_product' : None, 'min_product' : None }
        self.initialized = True


    def set_factors(self, factor_potentials, inputs, **kwargs):
        ''' Set the potential factors of this model, these includes all evidence factors etc.
            Arguments:
                factor_potentials: The list of PotentialTables which are the factors comprising this model
         '''
        if (not self.initialized):
            raise "Call alloc_resources first"
        pinputs = list(inputs)
        for i, v in enumerate(inputs):
            if (isinstance(v, PotentialTable)):
                pinputs[i] = v.pt_tensor
        self.pass_message_functions = {}
        self.initial_potential_expressions = self._create_initial_potential_expressions(factor_potentials)
        self._reset = self._create_reset_function(pinputs, **kwargs)
        self.compile_kwargs = kwargs

    def inference(self, input, threshold=0.001, verbose=True, max_iters=20, algo='sum_product', message_list=None, scheduler=simple_message_scheduler):
        scheduler(self, input, threshold, verbose, max_iters, algo, message_list)

    def reset(self, inputs):
        self._reset(*inputs)

    def pass_message(self, from_idx, to_idx, algo='sum_product'):
        ''' Pass a single message from clique from_idx to clique to_idx
            Should be used iteratively after a call to reset(...)
        '''
        if (algo not in self.pass_message_functions):
            self.pass_message_functions[algo] = self._create_pass_message_functions( algo, self.compile_kwargs)
        return self.pass_message_functions[algo][from_idx][to_idx]()

    def _create_pass_message_functions(self, algo, kwargs):
        result = []
        for i in range(self.clique_edges.shape[0]):
            row = []
            for j in range(self.clique_edges.shape[1]):
                if (self.clique_edges[i,j]==0):
                    row.append(None)
                else:
                    row.append(self._create_pass_message_fn(i, j, algo, kwargs))
            result.append(row)
        return result


    def _create_reset_function(self, input=[], **kwargs):
        ''' Returns a compiled theano function which resets the model to an initial (pre-inference) state by clearing all messages,
            and setting all clique potentials to their initial values (given the inputs)
            Arguments:
                input: Array of inputs which may be required to calculate the factors  (set by set_factors method)
                        and consequently the initial clique potentials set
                kwargs: Additional keyword arguments are passed to theano.function unmodified
            Returns:
                Compiled theano function
            '''
        assert self.initial_potential_expressions is not None
        assert len(self.initial_potential_expressions)==len(self.initial_potentials)
        update_list = list(zip([p.pt_tensor for p in self.initial_potentials],[p.pt_tensor for p in self.initial_potential_expressions]))
        nfloat = 1.0
        if (self.neutral=='zeros'):
            nfloat = 0.0
        reset_messages = self.shared_messages.reset(nfloat)
        update_list.append((self.shared_messages.message_mem, reset_messages))
        return theano.function(input, [], updates=update_list, on_unused_input='ignore', mode='DebugMode', **kwargs)

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

    def _create_initial_potential_expressions(self, factors=[]):
        '''
        Assign factors to cliques, and calculate initial potentials by multiplying these initial factors.
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

    def calc_a_message_order(self):
        ''' Given the clique list and their adjacency matrix, calculate a message passing order.
            Note: This message passing order is not neccessarily optimal in a loopy graph
        '''
        unmessaged = np.copy(self.clique_edges)
        message_ordering = list()
        reverse_messages = list()
        while True:
            next_cliques = np.nonzero(np.sum(unmessaged, axis=1)==1)[0] # Find all cliques which can pass a message immediately
            if (len(next_cliques)==0):
                break
            for src_clique in next_cliques:
                target_cliques = np.nonzero(unmessaged[:,src_clique])[0] # Find index of the one unmessaged neighbour
                if (len(target_cliques)<1):
                    break # This happens when this is the last clique remaining
                target_clique = target_cliques[0]
                message_ordering.append((src_clique, target_clique))
                reverse_messages.append((target_clique, src_clique))

                unmessaged[src_clique,target_clique]=0
                unmessaged[target_clique, src_clique]=0
        assert np.all(unmessaged==0)
        res = list(message_ordering) + list(reversed(reverse_messages))
        return res

    def _create_pass_message_fn(self, src_idx, target_idx, algo='sum_product', kwargs={}):
        return self.shared_messages.set_message_function(src_idx, target_idx, self._single_message_expr(src_idx, target_idx, algo), [], True, **kwargs)

    def _single_message_expr(self, src_idx, target_idx, algo='sum_product'):
        ''' Calculate theano expression for a single message from a clique to another
            using sum_product, max_product or min_product algorithms

            Args:
                src_idx index of source clique
                target_idx index of target_clique
                algo: May be either 'sum_product', 'max_product' or 'min_product'

            Returns:
                Theano expression which may be used to update the shared messages.
                This expression should be cached for later use.

            '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))

        init_potential = self.initial_potentials[src_idx]

        # We should cache this for repeated calls
        incoming_message_indices = np.nonzero(self.clique_edges[:,src_idx])[0]

        message_scope = self.clique_scopes[src_idx] & self.clique_scopes[target_idx]
        assert len(message_scope)>0
        message = init_potential

        for idx in incoming_message_indices:
            if (idx==src_idx or idx==target_idx):
                continue # Never pass a message back on itself.
            message = self.op(message, self.shared_messages.get_message_potential(idx, src_idx))
            if (algo=='sum_product'):
                if (self.logspace):
                    message = message.logsumexp_marginalize(message.scope - message_scope)
                else:
                    message = message.marginalize(message.scope - message_scope)
            if (algo=='max_product'):
                message = message.max_marginalize(message.scope - message_scope)
            if (algo=='min_product'):
                message = message.min_marginalize(message.scope - message_scope)

        return message

    def _current_clique_potential(self, clique_idx, algo='sum_product'):
        ''' Calculate theano expression for a single clique, which incorporates
            all incoming messages using sum_product, max_product or min_product algorithms

            Args:
                clique_idx index of  clique
                algo: May be either 'sum_product', 'max_product' or 'min_product'

            Returns:
                Theano expression of the clique's potential

            '''
        if (algo not in ['sum_product', 'max_product', 'min_product']):
            raise Exception("Invalid argument '%s' for algo parameter" % (algo))

        init_potential = self.initial_potentials[clique_idx]

        # We should cache this for repeated calls
        incoming_message_indices = np.nonzero(self.clique_edges[:,clique_idx])[0]

        scope = self.clique_scopes[clique_idx]
        assert len(scope)>0
        pot = init_potential

        for idx in incoming_message_indices:
            if (idx==clique_idx):
                continue # Never pass a message back on itself.
            pot = self.op(pot, self.shared_messages.get_message_potential(idx, clique_idx))
            if (algo=='sum_product'):
                if (self.logspace):
                    pot = pot.logsumexp_marginalize(pot.scope - scope)
                else:
                    pot = pot.marginalize(pot.scope - scope)
            if (algo=='max_product'):
                pot = pot.max_marginalize(pot.scope - scope)
            if (algo=='min_product'):
                pot = pot.min_marginalize(pot.scope - scope)

        return pot

    def _current_potentials(self, algo):
        ''' Return a list of PotentialTables containing symbolic tensor representations of the current
            clique potentials, which incorporate information from all incoming messages to the cliques.
            If the message passing has been run to convergence, potentials should agree on marginals
            and can be used for most relevant queries.
            Args:
                algo: May be either 'sum_product', 'max_product' or 'min_product'
            Returns:
                list of potential tables containing symbolic theano tensors representing the current cliques
            '''
        current_potentials = [self._current_clique_potential(ci, algo) for ci in range(len(self.clique_scopes))]
        return current_potentials

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
        calibrated = self._current_potentials('sum_product')
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



class SharedMessagePotentials(object):
    ''' Helper class, used by LoopyBPInference which represents the messages for loopy belief propagation as a single
        block of continuous memory. This shared memory can be updated using appropriate slices.

        It keeps track of message slice positions and brings the messages into appropriate shape.

        It allows for single and batch updating, reading and resetting of message states.
        '''


    def __init__(self, clique_scopes, discrete_pgm=None):
        if (discrete_pgm is None):
            discrete_pgm = DiscretePGM.get_context()

        self.discrete_pgm = discrete_pgm
        message_map = np.zeros((len(clique_scopes), len(clique_scopes)), dtype=np.int32)
        message_map[:] = -1
        self.message_scope_map = message_map
        self.message_scopes = []
        self.message_shapes = []
        for c1 in range(len(clique_scopes)):
            for c2 in range(len(clique_scopes)):
                if (c1==c2):
                    continue
                inters = clique_scopes[c1] & clique_scopes[c2]
                if (inters):
                    message_map[c1,c2] = len(self.message_scopes)
                    self.message_scopes.append(inters)
                    self.message_shapes.append(self._message_shape(c1, c2))

        self.message_slices = []
        off = 0
        for ms in self.message_scopes:
            message_size = 1
            for v in ms:
                message_size *= self.discrete_pgm.cardinalities[self.discrete_pgm.var_index(v)]
            self.message_slices.append(slice(off, off+message_size))
            off += message_size
        self.total_size = off
        self.clique_scopes = clique_scopes
        val = np.zeros((self.total_size), dtype=theano.config.floatX)
        self.message_mem = theano.shared(val, borrow=True)


    def _message_shape(self, from_idx, to_idx):
        ''' Returns the shape of the tensor of a message from from_idx to to_idx '''

        midx = self.message_scope_map[from_idx,to_idx]
        if (midx==-1):
            raise Exception("Message from %r to %r is not possible. They do not share an edge" % (from_idx, to_idx))
        var_set = self.message_scopes[midx]
        var_indices = sorted([ self.discrete_pgm.var_index(v) for v in var_set ])
        shp = []
        for vi in var_indices:
            shp.append(self.discrete_pgm.cardinalities[vi])
        return shp

    def message_potential_var(self, from_idx, to_idx, name=None, pt_tensor=None):
        midx = self.message_scope_map[from_idx,to_idx]
        if (midx==-1):
            raise Exception("Message from %r to %r is not possible. They do not share an edge" % (from_idx, to_idx))
        scope = self.message_scopes[midx]
        return PotentialTable(scope, pt_tensor, self.discrete_pgm, name=name)

    def get_message_potential(self, from_idx, to_idx, updated_message_mem=None):
        ''' Returns theano expression to read the message from from_idx to to_idx
            Allows to pass in an updated message_mem (for example, returned from the set_message method)
            in order to read intermediate results of the message memory
        '''
        midx = self.message_scope_map[from_idx,to_idx]
        if (midx==-1):
            raise Exception("Message from %r to %r is not possible. They do not share an edge" % (from_idx, to_idx))
        scope = self.message_scopes[midx]
        slic = self.message_slices[midx]
        shp = self.message_shapes[midx]
        message_mem = updated_message_mem
        if (message_mem is None):
            message_mem = self.message_mem
        subt = message_mem[slic]
        t = T.reshape(subt, shp)
        return PotentialTable(scope, t, self.discrete_pgm)

    def get_message_function(self, from_idx, to_idx, inputs=[],  *eargs, **enargs):
        '''
        Compiles and returns a function which returns the message from from_idx to to_idx.
        The function allows to pass inputs to the function.
        '''
        pt = self.get_message_potential(from_idx, to_idx)
        fn = theano.function(inputs, pt.pt_tensor, *eargs, **enargs)
        return fn

    def set_message_potential(self, from_idx, to_idx, message_potential, prev_target=None):
        '''
        Returns theano expression for the complete message mem, where the message from clique from_idx to to_idx has
        been changed to message_potential.

        If you want to perform batch updates, you can pass in the result of a previous set operation via the prev_target
        parameter.
        '''
        midx = self.message_scope_map[from_idx,to_idx]
        if (midx==-1):
            raise Exception("Message from %r to %r is not possible. They do not share an edge" % (from_idx, to_idx))
        scope = self.message_scopes[midx]
        slic = self.message_slices[midx]
        flatshape = (slic.stop-slic.start)
        target = prev_target
        if (target is None):
            target = self.message_mem

        tassign = T.reshape(message_potential.pt_tensor, [flatshape], ndim=1)
        result = T.set_subtensor(target[slic], tassign)
        return result

    def reset(self, reset_value=1.0):
        ''' Returns theano expression which can be assigned to the current message_mem to reset it to the given value '''
        if (reset_value==1.0):
            return T.ones_like(self.message_mem)
        if (reset_value==0.0):
            return T.zeros_like(self.message_mem)
        return T.fill(self.message_mem, np.array([reset_value], dtype=theano.config.floatX))
        #raise Exception("Reset only to 1.0 or 0.0 allowed")

    def reset_function(self, reset_value=1.0, *eargs, **enargs):
        ''' Compiles and returns a function which resets all values of the shared message state to the given reset_value,
        which should be 0 if working in log-space or 1 if working in probability-space
        '''
        reset_op = self.reset(reset_value)
        fun = theano.function([], [], updates=[(self.message_mem, reset_op)], *eargs, **enargs)
        return fun

    def set_message_function(self, from_idx, to_idx, message_potential, input=[], return_diff_sum=False, **enargs):
        ''' Compiles and returns a function which does the inplace update of the given messaging operation

            Arguments:
                from_idx: Index of the source clique
                to_idx: Index of the target clique
                message_potential: The PotentialTable of the calculated message
                input: Possible inputs to the function, which may be needed for the computation of the message_potential
            Returns:
                compiled function
        '''
        op = self.set_message_potential(from_idx, to_idx, message_potential)
        if (return_diff_sum):
            d1 = self.get_message_potential(from_idx, to_idx, self.message_mem)
            d2 = self.get_message_potential(from_idx, to_idx, op)

            change = d1.pt_tensor-d2.pt_tensor
            diffsum = T.sum(change.flatten())
            fun = theano.function(input, diffsum, updates=[(self.message_mem, op)], on_unused_input='ignore', **enargs)
        else:
            fun = theano.function(input, [], updates=[(self.message_mem, op)], on_unused_input='ignore', **enargs)
        return fun



