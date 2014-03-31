# PyPGMc

PyPGMc will provide a PyMC2 and PyMC3 compatible implementation of message passing based inference
algorithms for probabilistic graphical models (PGM).

The goal is to scale PyMC models up, and to enable some novel applications in the
powerful PyMC MCMC Framework.

### Warning

PyPGMc is under heavy development. It is by no means complete or bug-free yet. The code might serve as
example code, but that's it so far.

### License

PyPGMC is licensed under the Apache Public License 2.0 - see LICENSE file for details.

### Author & Copyright

Copyright: Kai Londenberg ( Kai.Londenberg@googlemail.com ) 2014

If you consider this interesting, you may contact me via email me or via LinkedIn:
[Kai Londenberg's LinkedIn Profile](http://de.linkedin.com/in/kailondenberg)

## Roadmap

The most important next steps are:

#### Exact discrete inference submodels in PyMC3.

I have afinished Clique Tree implementation using Theano which should be easily integrated with PyMC3. Being able to calculate exact derivatives through the model might allow for some pretty interesting applications. These clique trees would be appear as a discrete multivariate distribution to PyMC. 

#### Fast and scalable approximate inference submodels using Loopy BP for PyMC 2

While loopy BP isn't unbiased, it's an extremely fast and scalable general purpose inference algorithm. This is harder to integrate with PyMC3 due to the iterative nature of the algorithm (run it until convergence) it's less efficient to calculate derivatives etc. through the model. This is also pretty much implemented, but needs to be integrated with PyMC 2/3 and better tested and benchmarked.

#### Loader for at least one common Bayesian Network format

I have XDSL in mind, which has the advantage of being supported by the free BN GUI Tool GeNie, and the free (but closed source) SMILE Library which I already wrote a Python wrapper for (http://genie.sis.pitt.edu/). I will probably use that wrapper library at first, then later add own code to parse the format.

#### Support for (possibly incomplete) evidence for a discrete Bayes-Net in the form of a Pandas DataFrame. 

The BN would calculate the probability of the dataset given the hyperparameters. The hyperparameters themselves would be sampled by PyMC (for example from Dirichlet Distributions). This would obviously be very convenient. I'm not yet sure how to combine discrete and continuous evidence. Maybe some kind of Kabuki integration might help.

#### Examples, Examples !

I would like to show off some of the possibilities all of the above offers, probably in a few IPython Notebooks, tackling some old problems with new tools. 

  
### Extended Roadmap

All of the above would be pretty neat, and I think that will keep me occupied for a while. But then I have a few more ideas (which could probably keep me busy for years, and no I don't actually believe I will actually finish these, but hey ..)



 
#### More than just Dirichlet Priors ...

Once we have the above machinery in place, maybe we can explicity support some interesting priors over discrete distributions, such as a truncated Dirichlet Process Prior , or a smoothed or rank-ordered Dirichlet prior or things like that. This is probably easiest in PyMC3 for the exact inference engine.

#### Particle Message Passing

Message Passing can be extended to arbitrary non-discrete distributions by using particle-list based messages (see (http://www.dauwels.com/files/Particle.pdf) for a nice overview). What's interesting here: These particles could be easily created using MCMC submodels and combined with exact discrete messages. This would probably allow for inference using pretty much arbitrary mixed type models. Efficiency gains over pure MCMC are only to be expected if a substantial part of the model is discrete and can use exact messages, though.

I have my doubt whether this can be implemented easily enough in Theano, so it's probably something I would attempt for PyMC 2 first. 

#### Expectation Maximization Learning

It might very well be that for some variables we are not interested in unbiased samples, rather we would like to find a discrete set of local optima. An application for this might be Expectation Maximization (EM) learning over latent variables. The fact that we can get derivatives for theano-based PyMC3 models means we can use gradient-ascent for this
for almost arbitrary models.

#### Importance Sampling and Simulated Annealing

PyMC usually draws unweighted samples. I imagine it should be easy to add the ability to draw weighted samples from a modified (more or less peaked) distribution. In applications where risk is being modeled, it might be important to explore low probability density regions thoroughly. In estimation or learning, we might be more interested in the peaks. Allowing PyMC to have a variable "temperature" (as in Simulated Annealing) as well as record sample weights, the sampling efficiency for different purposes might be increased a lot, and PyMC might also be used for optimization purposes, more or less, out of the box. The ability to perform importance sampling is probably also essential for
particle based message passing, since these particles will usually be weighted.


#### Network Structure Learning

If we have a BN engine, it would probably make sense to include support for PEBL ( https://code.google.com/p/pebl-project/  ) which allows bayesian network structure learning, and maybe more explicit support for structure sampling like the one mentioned in this thread: https://groups.google.com/forum/#!topic/pymc/acTuyT4cp1Q - What might also be interesting is to implement causal structure learning algorithms like PC, IC and IC* (see Judea Pearls Book: Causality). Or we could leave this to existing tools like GeNie / SMILE etc..
