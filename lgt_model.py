"""Local and Global trend model"""

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.contrib.control_flow import scan

def model(y, future=0):
    T = len(y)
    cauchy_sd = jnp.max(y) / 150

    df = numpyro.sample('df', dist.Uniform(2, 20))
    rho = 1.5 * numpyro.sample('rho', dist.Beta(1., 1.)) - .5
    lamda = numpyro.sample('lamda', dist.Beta(1, 1))
    alpha = numpyro.sample('alpha', dist.Beta(1, 1))
    beta = numpyro.sample('beta', dist.Beta(1, 1))
    tau = numpyro.sample('tau', dist.Beta(1,1))
    xi = numpyro.sample('xi', dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(cauchy_sd))
    gamma = numpyro.sample('gamma', dist.Cauchy(0, cauchy_sd))

    def _transition(carry, i):
        level, trend = carry

        y_hat = level + gamma * level**rho + lamda * trend
        y_hat = jnp.clip(y_hat, a_min=0.)
        #use expected value when forecasting
        y_i = jnp.where(i > T, y_hat, y[i])

        _level = level
        level = alpha * y_i + (1-alpha) * _level
        level = jnp.clip(level, a_min=0.)

        trend = beta * (level-_level) + (1-beta) * trend

        sigma_hat = sigma * y_hat**tau + xi

        y_ = numpyro.sample('obs', dist.StudentT(df, y_hat, sigma_hat))
        return (level, trend), y_

    init = y[0]
    with numpyro.handlers.condition(data={'obs': y[1:]}):
        _, Y = scan(_transition, (init, init), jnp.arange(1, T+future))
    
def guide(y, future=0):
    # numpyro.sample('df', dist.Uniform(2, 20))
    with numpyro.handlers.reparam(config={'df': TransformReparam()}):
        numpyro.sample(
            'df',
            dist.TransformedDistribution(
                dist.Beta(numpyro.param('adf', 1., constraint=dist.constraints.positive),
                          numpyro.param('bdf', 1., constriant=dist.constraints.positive)),
                dist.transforms.AffineTransform(2., 18.)
            )
        )
    numpyro.sample(
        'rho', 
        dist.Beta(numpyro.param('arho', 1., constraint=dist.constraints.positive), 
                  numpyro.param('brho', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'lamda', 
        dist.Beta(numpyro.param('alamda', 1., constraint=dist.constraints.positive), 
                  numpyro.param('blamda', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'alpha', 
        dist.Beta(numpyro.param('aalpha', 1., constraint=dist.constraints.positive), 
                  numpyro.param('balpha', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'beta', 
        dist.Beta(numpyro.param('abeta', 1., constraint=dist.constraints.positive), 
                  numpyro.param('bbeta', 1., constrain=dist.constraints.positive))
    )
    numpyro.sample(
        'tau', 
        dist.Beta(numpyro.param('atau', 1., constraint=dist.constraints.positive), 
                  numpyro.param('btau', 1., constrain=dist.constraints.positive))
    )
    # numpyro.sample(
    #     'xi', 
    #     dist.TruncatedCauchy(low=1e-10, 
    #                          loc=numpyro.param('lxi', 1e-10, constraint=dist.constraints.greater_than(1e-10)), 
    #                          scale=numpyro.param('sxi', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'xi',
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('lxi', 0.), 
                        numpyro.param('sxi', 1., constraint=dist.constraints.positive)),
            dist.transforms.ExpTransform()
        )
    )
    # numpyro.sample(
    #     'sigma', 
    #     dist.HalfCauchy(numpyro.param('ssd', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'sigma',
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('msigma', 0.),
                        numpyro.param('ssigma', 1., constraint=dist.constraints.positive)),
            dist.transforms.ExpTransform()
        )
    )
    # numpyro.sample(
    #     'gamma', 
    #     dist.Cauchy(numpyro.param('mgamma', 0.),
    #                 numpyro.param('sgamma', 1., constraint=dist.constraints.positive))
    # )
    numpyro.sample(
        'gamma',
        dist.Normal(numpyro.param('mgamma', 0.),
                    numpyro.param('sgamma', 1., constraint=dist.constraints.positive))
    )