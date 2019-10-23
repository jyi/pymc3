import pymc3 as pm

h = 61

with pm.Model() as model:
    # prior
    n = pm.DiscreteUniform('n', lower=0, upper=300)
    # observed, likelihood
    y = pm.Binomial('y', n=n, p=0.61, observed=h)
    trace = pm.sample(5000, cores=1, chains=1, step=pm.Metropolis())

    print('RVs: {}'.format(model.basic_RVs))
    print('free RVs: {}'.format(model.free_RVs))
    print('observed RVs: {}'.format(model.observed_RVs))

# infer the value of n
# n is likely to be 100 since Binomial(100, 0.61) = 61
print(pm.summary(trace))
