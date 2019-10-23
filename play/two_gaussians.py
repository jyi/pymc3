import pymc3 as pm
import numpy as np
import theano.tensor as tt


chains = 500

n = 4

mu1 = np.ones(n) * (1. / 2)
mu2 = -mu1

stdev = 0.1
sigma = np.power(stdev, 2) * np.eye(n)
isigma = np.linalg.inv(sigma)
dsigma = np.linalg.det(sigma)

w1 = 0.1
w2 = (1 - w1)


def two_gaussians(x):
    log_like1 = - 0.5 * n * tt.log(2 * np.pi) \
        - 0.5 * tt.log(dsigma) \
        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
    log_like2 = - 0.5 * n * tt.log(2 * np.pi) \
        - 0.5 * tt.log(dsigma) \
        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
    return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))


with pm.Model() as ATMIP_test:
    X = pm.Uniform('X',
                   shape=n,
                   lower=-2. * np.ones_like(mu1),
                   upper=2. * np.ones_like(mu1),
                   testval=-1. * np.ones_like(mu1))
    llk = pm.Potential('llk', two_gaussians(X))


with ATMIP_test:
    trace = pm.sample(5000, chains=chains, step=pm.SMC())

print(pm.summary(trace))
