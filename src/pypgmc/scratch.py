import numpy as np
import theano
import theano.tensor as T
import theano.scan_module as TS


up_to = T.iscalar("up_to")

# define a named function, rather than using lambda
def accumulate_by_adding(arange_val, pot_val, sum_to_date):
    return sum_to_date + pot_val, TS.until(sum_to_date+pot_val>50)
seq = T.arange(up_to)
seq2 = T.arange(up_to)**2

# An unauthorized implicit downcast from the dtype of 'seq', to that of
# 'T.as_tensor_variable(0)' which is of dtype 'int8' by default would occur
# if this instruction were to be used instead of the next one:
# outputs_info = T.as_tensor_variable(0)

outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                        outputs_info=outputs_info,
                                        sequences=(seq, seq2))
triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result, mode='DebugMode')

theano.printing.debugprint(triangular_sequence.maker.fgraph.outputs[0], 20)


# test
some_num = 15
print triangular_sequence(some_num)
print [n * (n + 1) // 2 for n in xrange(some_num)]