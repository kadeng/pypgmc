'''

@author: londenberg
'''

from numpy.testing import assert_almost_equal
import numpy as np

def close_to(x, v, bound, name="value"):
    assert np.all(np.logical_or(
        np.abs(x - v) < bound,
        x == v)), name + " out of bounds : " + repr(x) + " and " + repr(v) + ", " + repr(bound)

class Context(object):
    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls, "contexts"):
            cls.contexts = []

        return cls.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")