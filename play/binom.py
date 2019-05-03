import pymc3 as pm

h = 61

with pm.Model():
    n = pm.DiscreteUniform('n', lower=0, upper=300)
    y = pm.Binomial('y', n=n, p=0.61, observed=h)
    trace = pm.sample(5000, cores=1, chains=1, step=pm.Metropolis())

print(pm.summary(trace));
