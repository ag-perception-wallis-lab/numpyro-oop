# An object-oriented interface to numpyro

This package provides a wrapper for working with [numpyro](https://num.pyro.ai/) models.
It aims to remain model-agnostic, but package up a lot of the model fitting code to reduce repetition.

## Getting started


### A basic demo

### TODO before initial release

- [x] group variable plate notation
- [x] reparameterisation dictionary
- [ ] arviz data export

### Roadmap after initial release

- [ ] better wrapping for `generate_arviz_data` (less user intervention)
- [ ] demo and tests for multiple group variables
- [ ] include doctest, improved examples
- [ ] export docs to some static page (readthedocs or similar); detail info on class methods and attributes
- [ ] CI test setup
- [ ] Contributor guidelines
- [ ] Fix type hints via linter checks


### Development notes

- Update dependencies with `make update-deps`
- Update and (re)install the environment with `make update-and-install`



