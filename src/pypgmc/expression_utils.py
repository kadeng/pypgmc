'''
@author: londenberg
'''

import theano.tensor as T

def mul_op(a, b):
    return a*b

def add_op(a,b):
    return a+b

def LogSumExp(x, axis=None, keepdims=True):
    ''' Numerically stable theano version of the Log-Sum-Exp trick'''
    x_max = T.max(x, axis=axis, keepdims=True)

    preres = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=keepdims))
    return preres + x_max.reshape(preres.shape)
