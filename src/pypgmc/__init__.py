'''

Exact Inference Submodels for PyMC3

Initial Implementation providing exact Clique Tree Inference for Bayesian Networks

To Do:
   - Ease integration of evidence into the clique tree
   - Use the scan-op to calculate aggregated probability of evidence
   - Test MAP-Calibration
   - Add sampling given incomplete evidence
   - Integrate into PyMC3 as a discrete probability distribution

Further Ideas:
   - Implement Dynamic Bayesian Network Inference
     see http://www.cs.ubc.ca/~murphyk/Thesis/thesis.html for details on the "Frontier" and "Interface" Algorithms

  -  Evaluate the Paper "A Differential Approach to Inference in Bayesian Networks" -> http://arxiv.org/abs/1301.3847
     for some new approaches. The approach I'm taking here using Theano is essentially an implementation of the ideas in that Paper
  -  Implement methods for cutting a Bayesian Network which is impossible to evaluate exactly into sub-networks which can be evaluated efficiently


Created on May 24, 2013
@author: Kai Londenberg (Kai.Londenberg@googlemail.com)
'''


from discrete_pgm import DiscretePGM
from potential_tables import PotentialTable
from clique_tree_inference import CliqueTreeInference
from loopy_bp_inference import LoopyBPInference, SharedMessagePotentials

