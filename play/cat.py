import sys
sys.path = ["/Users/jlee/workspace/angelix/build/pymc3/"] + sys.path

import numpy as np
import numpy.random as nr
import pymc3 as pm

bits = 2
max_num = np.power(2, bits)-1
start = {'p': 1} # represents 01

# markov chain
mc = 

def accept_fun(q, q0):
    return 0.5

class CustomProposal:
    def __init__(self, s):
        self.s = s
        
    def __call__(self, q0):
        q0 = q0.astype('int64')
        q = np.random.choice(np.arange(max_num+1),
                             1, p=[0.2, 0.5, 0.2, 0.1])
        import pdb; pdb.set_trace() # TODO: remove        
        return q

def proposal_dist(S):
    return nr.normal(scale=S)

with pm.Model() as model:
    p = pm.DiscreteUniform('p', 0, max_num)
    # s = pm.Categorical('s', p=p, shape=K)
    trace = pm.sample(2000, cores=1, chains=1,
                      start=start,
                      step=pm.Metropolis(accept_fun=accept_fun,
                                         S=mc,
                                         proposal_dist=CustomProposal,
                                         random_walk=True))

print(pm.summary(trace));
