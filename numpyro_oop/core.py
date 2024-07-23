from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from jax import random
from numpyro.infer import MCMC, NUTS, Predictive


class AbstractNumpyroModel(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def model(self, data: Optional[Any] = None):
        pass


class BaseNumpyroModel(AbstractNumpyroModel):
    """
    A BaseNumpyroModel provides the basic interface to numpyro-oop.


    _extended_summary_

    :param int seed: Random seed
    :param data: Data for the model. Could be e.g. a Pandas dataframe. The model
        method is expected to know what to do with data. Defaults to None.
    :param str kernel_type: Specify the type of MCMC kernel.
        Currently only "nuts" is supported.
    :param Dict kernel_kwargs: Keyword arguments passed to the MCMC kernel method.
    """

    def __init__(
        self,
        seed: int,
        data: Optional[Any] = None,
        kernel_type: str = "nuts",
        kernel_kwargs: Dict = {},
    ) -> None:
        if data is not None:
            self.data = data
        else:
            self.data = None

        self.rng_key = random.key(seed)

        if kernel_type.lower() == "nuts":
            kernel = NUTS(self.model, **kernel_kwargs)
        else:
            raise NotImplementedError("Only the NUTS kernel is currently implemented.")
        self.kernel = kernel

    def sample(
        self,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        num_chains: int = 4,
        model_kwargs: Dict = {},
        mcmc_kwargs: Dict = {},
    ):
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
        :param Dict model_kwargs: Keyword arguments passed to the model.
        :param Dict mcmc_kwargs: Keyword arguments passed to the MCMC object.
            See https://num.pyro.ai/en/stable/mcmc.html.
        """

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
        posterior_samples = self.mcmc.get_samples()
        self.posterior_samples = posterior_samples

    def predict(
        self,
        data: Optional[Any] = None,
        prior: bool = False,
        num_samples=200,
        model_kwargs: dict = {},
        predictive_kwargs: dict = {},
    ) -> Dict:
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
        :param int num_samples: The number of samples to generate.
        :param Dict model_kwargs: Keyword arguments passed to the model.
        :param Dict predictive_kwargs: Keyword arguments passed to the Predictive class.

        :return Dict: A dictionary containing samples from the predictive distribution.
        """
        if data is None:
            data = self.data
        if prior:
            posterior_samples = None
        else:
            posterior_samples = self.posterior_samples

        predictive = Predictive(
            self.model,
            num_samples=num_samples,  # ignored if posterior_samples is not None
            posterior_samples=posterior_samples,
            **predictive_kwargs,
        )

        self.rng_key, sub_key = random.split(self.rng_key)
        samples = predictive(sub_key, data=data, **model_kwargs)
        return samples
