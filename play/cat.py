import numpy as np
import numpy.random as nr
import pymc3 as pm

bits = 2
max_num = np.power(2, bits)-1
start = {'p': [0,1] }   # represents 00 and 01 (note that bit = 2)

"""
Construct a markov chain
"""
mc = np.array([[0.31, 0.3,  0.3,  0.09],
               [0.3,  0.31, 0.09, 0.3],
               [0.3,  0.09, 0.31, 0.3],
               [0.09, 0.3,  0.3,  0.31]])

def accept_fun(q, q0):
    return 0.5

class CustomProposal:
    """
    s: markov chain
    """
    def __init__(self, s):
        self.mc = s

    """
    q0: the current value
    return: proposed value
    """
    def __call__(self, q0):
        def propose(q0_i):
            q_i = np.random.choice(max_num+1, p=self.mc[q0_i])
            return q_i

        q0 = q0.astype('int64')
        q = list(map(propose, q0))
        return q

###################################################################

# import pdb; pdb.set_trace() # TODO: remove
with pm.Model() as model:
    p = pm.DiscreteUniform('p', 0, max_num, shape=2)

    pm.Metropolis(accept_fun=accept_fun,
                  S=mc,
                  proposal_dist=CustomProposal,
                  random_walk_mc=True)

    trace = pm.sample(2000, cores=1, chains=1,
                      start=start,
                      step=pm.Metropolis(accept_fun=accept_fun,
                                         S=mc,
                                         # mc is passed to CustomProposal
                                         proposal_dist=CustomProposal,
                                         random_walk_mc=True))

print(pm.summary(trace));
