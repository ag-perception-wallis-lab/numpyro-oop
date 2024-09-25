# An object-oriented interface to numpyro

[![Pytest API Testing](https://github.com/ag-perception-wallis-lab/numpyro-oop/actions/workflows/pytest.yml/badge.svg)](https://github.com/ag-perception-wallis-lab/numpyro-oop/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/numpyro-oop.svg)](https://badge.fury.io/py/numpyro-oop)

This package provides a wrapper for working with [numpyro](https://num.pyro.ai/) models.
It aims to remain model-agnostic, but package up a lot of the model fitting code to reduce repetition.

It is intended to make life a bit easier for people who are already familiar with Numpyro and Bayesian modelling.
It is not intended to fulfil the same high-level wrapper role as packages such as [brms](https://paul-buerkner.github.io/brms/).
The user is still required to write the model.

## Getting started

```
pip install numpyro-oop
```

The basic idea is that the user defines a new class that inherits from `BaseNumpyroModel`, 
and defines (minimally) the model to be fit by overwriting the `model` method:

```python
from numpyro_oop import BaseNumpyroModel

class DemoModel(BaseNumpyroModel):
    def model(self, data=None):
        ...

m1 = DemoModel(data=df, seed=42)
```

Then all other sampling and prediction steps are handled by `numpyro-oop`, or related libraries (e.g. `arviz`):

```python
m1.sample()  # sample from the model
preds = m1.predict()  # generate model predictions for the dataset given at initialization, or pass a new dataset
m1.generate_arviz_data()  # generate an Arviz InferenceData object stored in self.arviz_data
```

A more complete demo can be found in `/scripts/demo_1.ipynb`.

### Roadmap after initial release

- [ ] include doctest, improved examples
- [ ] demo and tests for multiple group variables
- [ ] export docs to some static page (readthedocs or similar); detail info on class methods and attributes
- [ ] Contributor guidelines
- [ ] Fix type hints via linter checks


### Development notes

- Update dependencies with `make update-deps`
- Update and (re)install the environment with `make update-and-install`



