from pypgmc import *
import theano
import theano.tensor as T

import numpy as np
from pymc3_adapter import close_to

def create_simple_cpt(vars=["a","b"]):
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    cpt = None
    try:
        allvars = ["a", "b", "c"]
        xshape = [3] * len(vars)
        initval = np.ones(tuple(xshape), dtype=theano.config.floatX)
        initvar = theano.shared(initval)
        cpt = PotentialTable(vars, initvar)
        for a in range(2):
            for b in range(2):
                cpt.set_value_of_assignment({vars[0] : a, vars[1] : b}, 0.1*a+0.2*b)
    finally:
        theano.config.compute_test_value = ctv_backup
    return cpt

def test_potential_tables():
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    try:
        dmodel = DiscretePGM([3,3,3], ["a", "b", "c"])
        with dmodel:
            cpt = create_simple_cpt()
            norm_cpt = cpt.normalize("a", False)

            res = theano.function([], cpt.pt_tensor)
            expected= np.array([[[ 0. ], [ 0.2], [ 1. ]], [[ 0.1], [ 0.3], [ 1. ]], [[ 1. ], [ 1. ],   [ 1. ]]], dtype=theano.config.floatX)
            #close_to(res(), expected, 0.01)

            nres = theano.function([], norm_cpt.pt_tensor)

            expected= np.array([[[ 0.], [ 0.16666667],[ 0.83333333]], [[ 0.07142857], [ 0.21428571],  [ 0.71428571]], [[ 0.33333333],  [ 0.33333333],  [ 0.33333333]]], dtype=theano.config.floatX)
            #close_to(nres(), expected, 0.01)

            cpt.set_conditional_prob("a", {"b" : 0}, np.array([0.1,0.2,0.3], dtype=theano.config.floatX))
            cpt.set_conditional_prob("a", {"b" : 1}, np.array([0.4,0.5,0.6], dtype=theano.config.floatX))
            cpt.set_conditional_prob("a", {"b" : 2}, np.array([0.7,0.8,0.9], dtype=theano.config.floatX))
            normcpt = cpt.normalize('a', inplace=False)

            avalue = T.iscalar('avalue')
            bvalue = T.iscalar('bvalue')

            ass = cpt.get_value_of_assignment({'a' : avalue, 'b' : bvalue})
            nass = normcpt.get_value_of_assignment({'a' : avalue, 'b' : bvalue})

            afn = theano.function([avalue, bvalue], ass, on_unused_input='warn')
            nafn = theano.function([avalue, bvalue], nass, on_unused_input='warn')
            nall = theano.function([], normcpt.pt_tensor)
            #print nall()
            assert afn(0,0)==0.1
            assert afn(2,1)==0.6
            assert afn(1,2)==0.8
            close_to(nafn(0,0)+nafn(1,0)+nafn(2,0), 1.0, 0.001)
            close_to(nafn(0,1)+nafn(1,1)+nafn(2,1), 1.0, 0.001)
            ecpt = cpt.observe_evidence({'b' : 1}, False)
            ecpt = ecpt.observe_evidence({'a' : 2}, False)
            necpt = ecpt.normalize("b", inplace=False)
            efn = theano.function([], ecpt.pt_tensor)
            nefn = theano.function([], necpt.pt_tensor)
            #print "After Evidence"
            ev = efn()
            nev = nefn()
            #print nev
            print nev.shape
            print nev

            assert nev[2,1]==1.0
            assert ev[2,1]==0.6
            nev[2,1]=0.0
            ev[2,1]=0.0
            assert np.all(nev==0.0)
            assert np.all(ev==0.0)
            xev = ecpt*necpt
            xefn = theano.function([], xev.pt_tensor)
            xevv = xefn()
            #print xevv
            assert xevv[2,1]==0.6
            scpt = create_simple_cpt( ['b', 'c'])
            #print scpt.pt_tensor.broadcastable
            scpt.set_conditional_prob("b", {"c" : 0}, np.array([0.1,0.2,0.3], dtype=theano.config.floatX))
            scpt.set_conditional_prob("b", {"c" : 1}, np.array([0.4,0.5,0.6], dtype=theano.config.floatX))
            scpt.set_conditional_prob("b", {"c" : 2}, np.array([0.7,0.8,0.9], dtype=theano.config.floatX))
            dscpt = scpt * scpt
            dfn = theano.function([], dscpt.pt_tensor)
            dfnv = dfn()
            #print dfnv
            #print dfnv[0,0,0]
            assert (dfnv[1,1] == 0.25)
            close_to(dfnv[0,0], 0.01, 0.0001)
            assert (dfnv[2,2] == 0.81)
            ascpt = create_simple_cpt( ['a', 'b'])
            ascpt.set_conditional_prob("b", {"a" : 0}, np.array([0.1,0.2,0.3], dtype=theano.config.floatX))
            ascpt.set_conditional_prob("b", {"a" : 1}, np.array([0.4,0.5,0.6], dtype=theano.config.floatX))
            ascpt.set_conditional_prob("b", {"a" : 2}, np.array([0.7,0.8,0.9], dtype=theano.config.floatX))

            scptfn = theano.function([], scpt.pt_tensor)
            scptv = scptfn()

            ascptfn = theano.function([], ascpt.pt_tensor)
            ascptv = ascptfn()


            mulres = scpt*ascpt

            print "scptv=%r" % (scptv)

            print "ascptv=%r" % (ascptv)


            # Check potential multiplication

            mulfn = theano.function([], mulres.pt_tensor)
            mulv = mulfn()

            print "mulres=scptv*ascptv = %r"  % (mulv)

            ones = PotentialTable(["a", "b", "c"], "ones")
            mulres1 = mulres*ones

            mulfn1 = theano.function([], mulres1.pt_tensor)
            mulv1 = mulfn1()
            print "mulres*ones=%r" % (mulv1)
            onesfn = theano.function([], ones.pt_tensor)
            onesv = onesfn()
            print "ones=%r" % (onesv)

            close_to(mulv1, mulv, 0.0001)
            close_to(onesv, np.ones(mulv1.shape, dtype=mulv1.dtype), 0.00001)

            print mulv.shape
            x2 = ascptv*scptv
            print x2.shape

            correct_mulv = np.array([[[ 0.01,  0.04,  0.07],
                [ 0.04,  0.1 ,  0.16],
                [ 0.09,  0.18,  0.27]],

               [[ 0.04,  0.16,  0.28],
                [ 0.1 ,  0.25,  0.4 ],
                [ 0.18,  0.36,  0.54]],

               [[ 0.07,  0.28,  0.49],
                [ 0.16,  0.4 ,  0.64],
                [ 0.27,  0.54,  0.81]]])
            close_to(mulv, correct_mulv, 0.0001)

            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        close_to(mulv[a,b,c], ascptv[a,b]*scptv[b,c], 0.0001)

            # Now check marginalization
            bres = mulres.marginalize(["a", "c"])
            bfn = theano.function([], bres.pt_tensor)
            bv = bfn()

            # Do the marginalization via numpy
            csum = np.sum(correct_mulv, 0)
            csum = np.sum(csum, 1)
            print bv
            print csum
            close_to(bv,csum, 0.0001)

             # Check max_marginalization operation

            mres = mulres.max_marginalize(["a", "c"])
            mfn = theano.function([], mres.pt_tensor)
            mv = mfn()

            print correct_mulv
            cmax = np.max(correct_mulv, 2)
            cmax = np.max(cmax, 0)

            print mv
            print cmax
            close_to(mv,cmax, 0.0001)

            max = mres.max_marginalize(["b"])
            maxb = theano.function([], max.pt_tensor)()
            assert maxb == np.max(cmax)

            print maxb


    finally:
        theano.config.compute_test_value = ctv_backup
    return "OK"

def check_is_spanning_tree(edges):
    if (edges.shape[0]==1):
        return True # Just one clique
    # Diagonal should be empty
    assert np.all(np.diag(edges)==0)
    # Edge Matrix should be symmetric
    assert np.all(np.triu(edges)==np.transpose(np.tril(edges)))
    connected = np.sum(edges, axis=1)
    # Every Clique should be connected to some other
    assert np.all(connected>=1)

    visited = np.zeros((edges.shape[0]), dtype=np.int8)

    def visit(edges, visited, to_visit, coming_from, depth=1000):
        if (depth==0):
            raise Exception("Maximum recursion level reached when visiting graph")
        visited[to_visit] += 1
        assert visited[to_visit]==1
        for i in range(edges.shape[0]):
            if (i==to_visit or i==coming_from):
                continue
            if (edges[to_visit,i]==1):
                visit(edges, visited, i, to_visit, depth-1)

    visit(edges, visited, 0, -1, edges.shape[0]+2)
    assert np.all(visited==1)

def create_random_factors(numvars=100, numfactors=100, vars_per_factor=(2,5)):
    factor_scopes = set()
    while (len(factor_scopes)<numfactors):
        scopesize = np.random.randint(vars_per_factor[0],vars_per_factor[1]+1)
        scope = set()
        while (len(scope)<scopesize):
            var = np.random.randint(0, numvars)
            scope.add(int(var))
        factor_scopes.add(frozenset(scope))
    factor_scopes = sorted(list(factor_scopes))
    factor_edges = np.zeros((numfactors,numfactors), dtype=np.int8)
    for f1 in range(numfactors):
        for f2 in range(numfactors):
            if (factor_scopes[f1] & factor_scopes[f2]): # Share a common variable
                factor_edges[f1,f2] = 1

    # Now we return a random connected subset of all factors
    connectivity = np.sum(factor_edges, axis=1)
    start_candidates = np.nonzero(connectivity>3)[0]
    if (len(start_candidates)==0):
        start_candidates = np.nonzero(connectivity>2)[0]
    if (len(start_candidates)==0):
        start_candidates = np.nonzero(connectivity>1)[0]
    if (len(start_candidates)==0):
        # No candidates ? That might happen. Try again.
        return create_random_factors(numvars, numfactors, vars_per_factor)
    start_factor = start_candidates[np.random.randint(0, len(start_candidates))]
    result_factors = set([start_factor])

    def add_connected_factors(result_factors, factor, factor_edges, backlink=-1):
        for f in np.nonzero(factor_edges[:,factor])[0]:
            if (f not in result_factors):
                result_factors.add(int(f))
                add_connected_factors(result_factors, f, factor_edges, factor)

    add_connected_factors(result_factors, start_factor, factor_edges)
    result = [sorted(list(factor_scopes[f])) for f in result_factors]
    return result

def test_clique_tree_creation():
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    card = 5
    try:
        dmodel = DiscretePGM([card]*400)
        factors = []
        with dmodel:
            factors.append(PotentialTable([0])) # P(a)
            factors.append(PotentialTable([1,0])) # P(a|b)
            factors.append(PotentialTable([2,1,3])) # P(c|b,d)
            ctree = CliqueTreeInference(factors)
            print ctree.elimination_order
            assert ctree.elimination_order == [0,1,2,3]
            assert frozenset(ctree.clique_scopes) == frozenset([frozenset([1, 2, 3]), frozenset([0, 1])])
            assert np.all(ctree.clique_edges == [[0,1],[1,0]])


            check_is_spanning_tree(ctree.clique_edges)
            factors = [PotentialTable(f) for f in [[4, 7, 8], [2, 4, 5, 7, 8]]]
            ctree = CliqueTreeInference(factors)
            loopy = LoopyBPInference(factors)
            check_is_spanning_tree(ctree.clique_edges)
            print "Mem-Usage: Clique Tree: %d kb, Loopy BP: %d kb" % (ctree.get_mem_usage()/1024, (loopy.get_mem_usage()/1024))
            # Now do some hardcore randomized testing
            for nvars in [30, 50, 100]:
                for nfac in [0.33, 0.66, 1.0]:
                    factor_vars = create_random_factors(nvars, int(nvars*nfac))
                    vrs = set()
                    for f in factor_vars:
                        vrs.update(f)
                    factors = [PotentialTable(f) for f in factor_vars]

                    print "\tChecking Random Clique Tree with %d vars and %d factors" % (len(vrs), len(factors))
                    ctree = CliqueTreeInference(factors)
                    loopy = LoopyBPInference(factors)

                    check_is_spanning_tree(ctree.clique_edges)
                    ctree._calc_message_order() # This checks that all cliques have been used
                    vrs = list(vrs)
                    for rvar in vrs:
                        #random_var = vrs[np.random.randint(0, len(vrs))]
                        vindices = []
                        for i in range(len(ctree.clique_scopes)):
                            if (rvar in ctree.clique_scopes[i]):
                                vindices.append(i)
                        if (len(vindices)>2):
                            subtree_edges = ctree.clique_edges[vindices,:][:,vindices]
                            #print "\t\tChecking running intersection property for var %d in that clique tree (appearing in %d factors)" % (rvar, len(vindices))
                            check_is_spanning_tree(subtree_edges)
                            #print "\t\tOK"
                    print "\tOK - Clique Count: %d - max clique size: %d " % (len(ctree.clique_scopes), max([len(scp) for scp in ctree.clique_scopes]))
                    print "\tCardinality: %d - Mem-Usage: Clique-Tree: %d kb, Loopy BP: %d kb" % (card, ctree.get_mem_usage()/1024, (loopy.get_mem_usage()/1024))
                    print "Allocating loopy bp resources"
                    loopy.alloc_resources()
                    print "Done allocating"

            return "OK"

    finally:
        theano.config.compute_test_value = ctv_backup

def test_clique_tree_calibration():
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    try:
        dmodel = DiscretePGM([2,2,2,2],["a","b","c", "d"])
        factors = []
        with dmodel:
            va = np.random.random(size=[2,2]) + 0.001
            vb = np.random.random(size=[2]) + 0.001
            vc = np.random.random(size=[2,2]) + 0.001
            vd = np.random.random(size=[2,2]) + 0.001

            vas = theano.shared(va)
            vbs = theano.shared(vb)
            vcs = theano.shared(vc)
            vds = theano.shared(vd)

            c_evidence = PotentialTable(["c"], name="E(c)")

            factors.append(PotentialTable(["a","b"],vas,  name="P(a|b)").normalize("a", inplace=False))
            factors.append(PotentialTable(["b"],vbs, name="P(b)").normalize("b", inplace=False))
            factors.append(PotentialTable(["c","b"], vcs, name="P(c|b)").normalize("c", inplace=False))
            factors.append(PotentialTable(["d", "c"], vds, name="P(d|c)").normalize("d", inplace=False))
            factors.append(c_evidence)
    #
            ctree = CliqueTreeInference(factors)
            print ctree.clique_scopes
            print ctree.clique_edges

            probexpr = ctree.probability(factors)
            probfunc = theano.function([c_evidence.pt_tensor], probexpr, mode='DebugMode', on_unused_input='warn')

            evidence = np.ones((2), dtype=theano.config.floatX)
            prob = probfunc(evidence)
            print prob
            assert abs(probfunc(evidence)-1.0)<0.0001
            evidence[0] = 0
            p1 = probfunc(evidence)
            evidence[0] = 1.
            evidence[1] = 0.
            p2 = probfunc(evidence)
            assert abs(p1+p2-1.0)<0.0001

            equivalent_potential = factors[0]
            for i in range(1, len(factors)):
                equivalent_potential = equivalent_potential * factors[i]

            eprob = T.sum(equivalent_potential.pt_tensor, axis=[0,1,2,3])
            eprobfunc = theano.function([c_evidence.pt_tensor], eprob)

            evidence = np.ones((2))
            evidence[0] = 0
            ep1 = eprobfunc(evidence)
            evidence[0] = 1.
            evidence[1] = 0.
            ep2 = eprobfunc(evidence)
            assert abs(ep1+ep2-1.0)<0.0001
            assert abs(ep1-p1)<0.00001
            assert abs(ep2-p2)<0.00001

            return "OK"

    finally:
        theano.config.compute_test_value = ctv_backup

def test_log_clique_tree_calibration():
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    try:
        dmodel = DiscretePGM([2,2,2,2],["a","b","c", "d"])
        factors = []
        logfactors = []
        with dmodel:
            va = np.random.random(size=[2,2]) + 0.001
            vb = np.random.random(size=[2]) + 0.001
            vc = np.random.random(size=[2,2]) + 0.001
            vd = np.random.random(size=[2,2]) + 0.001

            vas = theano.shared(va)
            vbs = theano.shared(vb)
            vcs = theano.shared(vc)
            vds = theano.shared(vd)

            c_evidence = PotentialTable(["c"], name="E(c)")

            factors.append(PotentialTable(["a","b"],vas,  name="P(a|b)").normalize("a", inplace=False))
            factors.append(PotentialTable(["b"],vbs, name="P(b)").normalize("b", inplace=False))
            factors.append(PotentialTable(["c","b"], vcs, name="P(c|b)").normalize("c", inplace=False))
            factors.append(PotentialTable(["d", "c"], vds, name="P(d|c)").normalize("d", inplace=False))
            factors.append(c_evidence)

            logfactors = [f.to_logspace(inplace=False) for f in factors]

            lff = theano.function([c_evidence.pt_tensor], [l.pt_tensor for l in logfactors])


            ctree = CliqueTreeInference(logfactors, None, True)
            ctree2 = CliqueTreeInference(factors, None, False)

            probexpr = ctree.probability(logfactors)
            probfunc = theano.function([c_evidence.pt_tensor], probexpr,  on_unused_input='warn')

            probexpr2 = ctree2.probability(logfactors)
            probfunc2 = theano.function([c_evidence.pt_tensor], probexpr2,  on_unused_input='warn')

            evidence = np.ones((2), dtype=theano.config.floatX)

            p0 = probfunc(evidence)
            p02 = probfunc2(evidence)

            #assert abs(probfunc(evidence.reshape([1,1,2,1]))-1.0)<0.0001
            evidence[0] = 0
            p1 = probfunc(evidence)

            evidence[0] = 1.
            evidence[1] = 0.
            p2 = probfunc(evidence)

            print p1
            print p2

            assert abs(p1+p2-1.0)<0.0001

            equivalent_potential = factors[0]
            for i in range(1, len(factors)):
                equivalent_potential = equivalent_potential * factors[i]


            eprob = T.sum(equivalent_potential.pt_tensor, axis=[0,1,2,3])
            eprobfunc = theano.function([c_evidence.pt_tensor], eprob)

            evidence = np.ones((2))
            evidence[0] = 0
            ep1 = eprobfunc(evidence)
            evidence[0] = 1.
            evidence[1] = 0.
            ep2 = eprobfunc(evidence)
            assert abs(ep1+ep2-1.0)<0.0001
            assert abs(ep1-p1)<0.00001
            assert abs(ep2-p2)<0.00001


            #theano.printing.pydotprint(pevf, '/tmp/func.png', var_with_name_simple = True, with_ids = True)
            return "OK"

    finally:
        theano.config.compute_test_value = ctv_backup


def test_shared_message_potentials():
    ctv_backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'off'
    try:
        dmodel = DiscretePGM([2,2,2,2],["a","b","c", "d"])
        factors = []
        logfactors = []
        with dmodel:
            va = np.random.random(size=[2,2]) + 0.001
            vb = np.random.random(size=[2]) + 0.001
            vc = np.random.random(size=[2,2]) + 0.001
            vd = np.random.random(size=[2,2]) + 0.001

            vas = theano.shared(va)
            vbs = theano.shared(vb)
            vcs = theano.shared(vc)
            vds = theano.shared(vd)

            c_evidence = PotentialTable(["c"], name="E(c)")

            factors.append(PotentialTable(["a","b"],vas,  name="P(a|b)").normalize("a", inplace=False))
            factors.append(PotentialTable(["b"],vbs, name="P(b)").normalize("b", inplace=False))
            factors.append(PotentialTable(["c","b"], vcs, name="P(c|b)").normalize("c", inplace=False))
            factors.append(PotentialTable(["d", "c"], vds, name="P(d|c)").normalize("d", inplace=False))
            factors.append(c_evidence)

            logfactors = [f.to_logspace(inplace=False) for f in factors]

            ctree = CliqueTreeInference(logfactors, None, True)

            probexpr = ctree.probability(logfactors)
            probfunc = theano.function([c_evidence.pt_tensor], probexpr)

            evidence = np.ones((2))

            loopy = LoopyBPInference(logfactors, None, True)
            loopy.alloc_resources()
            assert set(ctree.clique_scopes) == set(loopy.clique_scopes)

            msgs = loopy.shared_messages
            print msgs._message_shape(0,1)
            print msgs._message_shape(2,0)
            print "_--"
            try:
                print msgs._message_shape(1,1)
                assert False
            except:
                pass
            try:
                print msgs._message_shape(1,2)
                assert False
            except:
                pass
            shmem = loopy.shared_messages.message_mem
            shape = theano.function([], [shmem.shape])

            print shape()
            reset_fn = msgs.reset_function(0.0)
            get_fn = msgs.get_message_function(0, 2)
            get_expr = msgs.get_message_potential(0,2)
            sval = msgs.message_potential_var(0, 2)
            set_fn = msgs.set_message_function(0, 2, sval, [sval.pt_tensor])
            reset_fn()
            zero_msg = get_fn()
            print get_fn().shape
            print zero_msg
            print "Expecting 1"
            print shmem.get_value()
            reset_fn = msgs.reset_function(1.0)
            reset_fn()
            ones_msg = np.array(get_fn())
            print ones_msg
            print "Expecting 0"
            print shmem.get_value()
            reset_fn = msgs.reset_function(2.0)
            print "Expecting 2"
            reset_fn()
            print shmem.get_value()
            print get_fn()
            print ones_msg.shape
            set_fn(ones_msg)
            print shmem.get_value()
        return "OK"
    finally:
        theano.config.compute_test_value = ctv_backup

if __name__ == '__main__':
    print "Testing Potential Table Class .. "
    print test_potential_tables()
    print "Testing Clique Tree Creation .. "
    print test_clique_tree_creation()
    print "Testing clique tree calibration"
    print test_clique_tree_calibration()
    print "Testing clique tree calibration in log space"
    print test_log_clique_tree_calibration()
    print "Testing SharedMessagePotentials class"
    test_shared_message_potentials()
