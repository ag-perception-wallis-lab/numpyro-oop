# An object-oriented interface to numpyro

[![CI](https://github.com/ag-perception-wallis-lab/numpyro-oop/actions/workflows/pytest.yml/badge.svg)](https://github.com/ag-perception-wallis-lab/numpyro-oop/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/numpyro-oop.svg)](https://badge.fury.io/py/numpyro-oop)

This package provides a wrapper for working with [numpyro](https://num.pyro.ai/) models.
It aims to remain model-agnostic, but package up a lot of the model fitting code to reduce repetition.

It is intended to make life a bit easier for people who are already familiar with Numpyro and Bayesian modelling.
It is not intended to fulfil the same high-level wrapper role as packages such as [brms](https://paul-buerkner.github.io/brms/).
The user is still required to write the model from scratch.
This is an intentional choice: writing the model from scratch takes longer and is less convenient for standard models like GLMs, but has the advantage that one gains a deeper insight into what is happening under the hood, and also it is more transparent to implement custom models that don't fit a standard mould.

## Getting started

```
pip install numpyro-oop
```

The basic idea is that the user defines a new class that inherits from `BaseNumpyroModel`, 
and defines (minimally) the model to be fit by overwriting the `model` method:

```python
from numpyro_oop import BaseNumpyroModel

class DemoModel(BaseNumpyroModel):
    def model(self, data=None, ...):
        ...

m1 = DemoModel(data=df, seed=42)
```

Then all other sampling and prediction steps are handled by `numpyro-oop`, or related libraries (e.g. `arviz`):

```python
# sample from the model:
m1.sample()  
# generate model predictions for the dataset given at initialization:
preds = m1.predict(...)
# generate an Arviz InferenceData object stored in self.arviz_data:
m1.generate_arviz_data()  
```
A complete demo can be found in `/scripts/demo_1.ipynb`.


## Requirements of the `model` method

Consider the following model method:

```python
class DemoModel(BaseNumpyroModel):
    def model(self, data=None, sample_conditional=True, ...):
        ...

        if sample_conditional:
            obs = data["y"].values
        else:
            obs = None
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)

m1 = DemoModel(data=df, seed=42)
```

First, note that we can pass `data` as an optional `kwarg` that defaults to `None`.
If data is not passed to the `model` object directly then
`model` will automatically fall back to `self.data`, defined when the class instance is initialised.

Second, note the `sample_conditional` argument and subsequent pattern.
To use Numpyro's `Predictive` method, we need the ability to set any observed data 
that a sampling distribution is conditioned upon (typically the likelihood) to be `None`.
See the [Numpyro docs](https://num.pyro.ai/en/stable/utilities.html#numpyro.infer.util.Predictive) for examples.
Currently, `numpyro-oop` requires this to be implemented by the user in the model definition 
in some way; a suggested pattern is shown above.

After the model is sampled, we can then generate posterior predictive distributions by 
passing `sample_conditional=False` as a `model_kwarg`:

```python
m1.predict(model_kwargs={"sample_conditional": False})
```

## Using reparameterizations

One of the really neat features of Numpyro is the ability to define reparameterizations 
of variables that can be applied to the model object 
(see [docs](https://num.pyro.ai/en/stable/handlers.html#numpyro.handlers.reparam)).
To use these with `numpyro-oop`, the user must overwrite the `generate_reparam_config`
method of `BaseNumpyroModel` to return a reparameterization dictionary:

```python
def generate_reparam_config(self) -> dict:
    reparam_config = {
        "theta": LocScaleReparam(0),
    }
    return reparam_config
```

In this example, the node `theta` in the model will be reparameterized with a location/scale reparam
if `use_reparam=True` when the class instance is created. 
This is handy, because you can then test the effect of your reparameterization by simply setting
`use_reparam=False` and re-fitting the model.
See `examples/demo.ipynb` for a 
working example.

## Roadmap after initial release

- [ ] include doctest, improved examples
- [ ] demo and tests for multiple group variables
- [ ] export docs to some static page (readthedocs or similar); detail info on class methods and attributes
- [ ] Contributor guidelines
- [ ] Fix type hints via linter checks


### Development notes

Install the library with development dependencies via `pip install -e ".[dev]"`.


