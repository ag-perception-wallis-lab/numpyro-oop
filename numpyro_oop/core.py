from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from jax import random
from numpyro import render_model
from numpyro.infer import MCMC, NUTS, Predictive

__all__ = ["BaseNumpyroModel", "SamplingKernelType"]


class SamplingKernelType(Enum):
    nuts = NUTS


class AbstractNumpyroModel(ABC):
    @abstractmethod
    def sample(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> dict:
        pass

    @abstractmethod
    def model(self, data: Optional[Any] = None) -> None:
        pass

    @abstractmethod
    def _model(self, data: Optional[Any] = None) -> None:
        pass

    @abstractmethod
    def render(self, kwargs: Optional[dict] = None):
        pass


class BaseNumpyroModel(AbstractNumpyroModel):
    """
    A BaseNumpyroModel provides the basic interface to numpyro-oop.

    :param int seed: Random seed
    :param data: Data for the model. Could be e.g. a Pandas dataframe. The model
        method is expected to know what to do with data. Defaults to None.
    """

    def __init__(
        self,
        seed: int,
        data: Optional[Any] = None,
    ) -> None:
        self.data = data
        self.rng_key = random.key(seed)

    def _model(
        self, data: Optional[Any] = None, model_kwargs: Optional[dict] = None
    ) -> None:
        """
        An internal model caller to perform runtime checks.

        :param data: data for the model; will default to self.data if None.
        :param model_kwargs: optional keyword arguments for the model.
        """
        if data is None:
            data = self.data
        model_kwargs = model_kwargs or {}
        self.model(data=data, **model_kwargs)

    def sample(
        self,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        num_chains: int = 4,
        kernel_type: Optional[SamplingKernelType] = SamplingKernelType.nuts,
        model_kwargs: Optional[dict] = None,
        kernel_kwargs: Optional[dict] = None,
        mcmc_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Draw MCMC samples from the model.

        Samples from the model using the
        kernel and data specified at instantiation.
        A wrapper around MCMC: https://num.pyro.ai/en/stable/mcmc.html.

        The MCMC object will be stored in the
        class instance in the `mcmc` attribute.
        Posterior samples will be stored in `posterior_samples`.

        :param int num_samples: Number of samples to draw from the Markov chain.
        :param int num_warmup: Number of warmup steps.
        :param int num_chains: Number of chains.
        :param str kernel_type: Specify the type of MCMC kernel.
            Currently only "nuts" is supported.
        :param dict model_kwargs: Keyword arguments passed to the model.
        :param Dict kernel_kwargs: Keyword arguments passed to the MCMC kernel method.
        :param dict mcmc_kwargs: Keyword arguments passed to the MCMC object.
            See https://num.pyro.ai/en/stable/mcmc.html.
        """

        kernel_kwargs = kernel_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.kernel = kernel_type.value(self._model, **kernel_kwargs)

        self.mcmc = MCMC(
            self.kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            **mcmc_kwargs,
        )

        # https://jax.readthedocs.io/en/latest/jax.random.html
        self.rng_key, sub_key = random.split(self.rng_key)
        self.mcmc.run(sub_key, data=self.data, **model_kwargs)
        self.posterior_samples = self.mcmc.get_samples()

    def predict(
        self,
        data: Optional[Any] = None,
        prior: bool = False,
        num_samples=200,
        model_kwargs: Optional[dict] = None,
        predictive_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Create a predictive distribution.

        This method is a wrapper around the Predictive
        class (https://num.pyro.ai/en/latest/utilities.html#predictive).
        It can be used to create predictive distributions for
        priors or posteriors, based on the data used to fit the model
        or using new data.

        :param Any data: The data to predict. If None, will use the data passed at initialisation.
            If new data is passed, a predictive distribution will be generated for this new data.
        :param bool prior: If True, generates prior predictive samples.
        :param int num_samples: The number of samples to generate. Due to an unexpected
            numpyro behaviour, this will be ignored if prior is False (will use size of
            posterior_samples).
        :param dict model_kwargs: Keyword arguments passed to the model.
        :param dict predictive_kwargs: Keyword arguments passed to the Predictive class.

        :return dict: A dictionary containing samples from the predictive distribution.
        """
        if data is None:
            data = self.data

        if prior:
            posterior_samples = None
        else:
            posterior_samples = self.posterior_samples

        model_kwargs = model_kwargs or {}
        predictive_kwargs = predictive_kwargs or {}

        predictive = Predictive(
            self._model,
            num_samples=num_samples,  # ignored if posterior_samples is not None
            posterior_samples=posterior_samples,
            **predictive_kwargs,
        )

        self.rng_key, sub_key = random.split(self.rng_key)
        samples = predictive(sub_key, data=data, **model_kwargs)
        return samples

    def render(
        self,
        render_distributions: bool = True,
        render_params: bool = True,
        kwargs: Optional[dict] = None,
    ):
        """
        Function to render the model graph

        Wrapper of https://num.pyro.ai/en/latest/utilities.html#numpyro.infer.inspect.render_model

        :param bool render_distributions: Should RV distributions annotations be included.
        :param bool render_params: Show params in the plot.
        :param optional kwargs: Keyword arguments passed to render_model.
        :return: The graphviz graph.
        """
        kwargs = kwargs or {}
        graph = render_model(
            self._model,
            render_distributions=render_distributions,
            render_params=render_params,
            **kwargs,
        )
        return graph
