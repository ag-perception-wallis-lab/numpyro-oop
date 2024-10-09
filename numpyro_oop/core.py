from logging import getLogger

logger = getLogger(__name__)


from enum import Enum
from typing import Optional

import arviz as az
import numpyro
import pandas as pd
from jax import random
from numpyro import render_model
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, Predictive

__all__ = ["BaseNumpyroModel", "SamplingKernelType"]


class SamplingKernelType(Enum):
    nuts = NUTS


class BaseNumpyroModel:
    def __init__(
        self,
        seed: int,
        data: Optional[pd.DataFrame] = None,
        group_variables: Optional[list[str] | str] = None,
        create_plates_kwargs: Optional[dict] = None,
        use_reparam: bool = True,
    ) -> None:
        """
        A BaseNumpyroModel provides the basic interface to numpyro-oop.

        :param int seed: Random seed
        :param data: Data for the model. Currently only support Pandas dataframes. The model
            method is expected to know what to do with data. Note that a copy of the passed
            dataframe is made to avoid unanticipated effects on the outer-scope dataframe.
        :param group_variables: Names of the variables in data that correspond to discrete
            categories to be used for plates in the model (see https://num.pyro.ai/en/stable/primitives.html#plate).
            A demo of plates can be found here: https://num.pyro.ai/en/stable/tutorials/bayesian_hierarchical_linear_regression.html.
            String or list of strings.
        :param create_plates_kwargs: Keyword arguments passed to the internal _create_plates function.
        :param bool use_reparam: If true, and the generate_reparam_config method returns a
            configuration dictionary, this will apply the reparameterization to the model.
            For this to have an effect, overwrite the
            generate_reparam_config method. https://num.pyro.ai/en/stable/handlers.html#reparam
        """
        if data is not None:
            self.data = data.copy()
        else:
            self.data = None
        self.group_variables = group_variables or []
        self.rng_key = random.key(seed)

        if self.group_variables:
            if type(self.group_variables) is str:
                logger.debug(
                    f"A single string was passed to group_variables; converting to list."
                )
                self.group_variables = [self.group_variables]
            create_plates_kwargs = create_plates_kwargs or {}
            self._create_plates(**create_plates_kwargs)
        else:
            self.plate_dicts = None

        reparam_config = self.generate_reparam_config()

        if use_reparam and reparam_config:
            self.model = reparam(self.model, config=reparam_config)

        self.posterior_samples = None
        self.posterior_predictive = None
        self.prior_predictive = None
        self.arviz_data = None

    def _model(
        self, data: Optional[pd.DataFrame] = None, model_kwargs: Optional[dict] = None
    ) -> None:
        """
        An internal model caller to perform runtime checks.

        :param data: data for the model; will default to self.data if None.
        :param model_kwargs: optional keyword arguments for the model.
        """
        if data is None:
            data = self.data
        model_kwargs = model_kwargs or {}
        logger.info(
            "\nThe following group variables are available for specifying plates: {self.group_variables}"
        )
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
        generate_arviz_data: Optional[bool] = False,
        generate_arviz_data_kwargs: Optional[dict] = None,
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
        :param bool generate_arviz_data: If True, generate arviz data and store in the
            model object. Equivalent to running self.generate_arviz_data.
        :param dict generate_arviz_data_kwargs: Keyword arguments passed to the
            generate_arviz_data method. If None, default arguments will be used.
        """

        kernel_kwargs = kernel_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}
        model_kwargs = model_kwargs or {}
        generate_arviz_data_kwargs = generate_arviz_data_kwargs or {}

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
        self.mcmc.run(sub_key, data=self.data, model_kwargs=model_kwargs)
        self.posterior_samples = self.mcmc.get_samples()

        if generate_arviz_data:
            self.generate_arviz_data(**generate_arviz_data_kwargs)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
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
        :param dict predictive_kwargs: Keyword arguments passed to Numpyro's Predictive class.

        :return dict: A dictionary containing samples from the predictive distribution.
        """
        if data is None:
            data = self.data
            new_data = False
        else:
            new_data = True

        if prior:
            posterior_samples = None
        else:
            if self.posterior_samples is None:
                raise ValueError(
                    (
                        "You tried to generate posterior predictions, but you haven't"
                        " sampled from the model yet! "
                        "You must first run the 'sample' method."
                    )
                )
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
        samples = predictive(sub_key, data=data, model_kwargs=model_kwargs)

        if not new_data:
            if prior:
                self.prior_predictive = samples
            else:
                self.posterior_predictive = samples

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

    @staticmethod
    def _cats_to_dict(x: pd.Series) -> dict:
        """
        Turn Pandas categorical categories into a dict.

        The dict is used for having an integer index to the
        group member. The returned dict will have integer keys
        and values equal to the categorical levels.

        The Pandas Series must contain data of type categorical.
        Can be set as .astype("category").

        :param x: A pandas series containing categorical data.
        """
        return dict(enumerate(x.cat.categories))

    def _create_plates(
        self, variable_suffix: str = "_id", subsample_size: Optional[int] = None
    ) -> None:
        """
        Create plates for model specification

        This method will modify self.data and create a new dict-of-dicts
        called self.plate_dicts. The plate_dicts contain information about
        the plate defined by a categorical group in the data.

        The modification to the data itself involves adding a new variable to the dataframe for
        each group_variable (appending suffix "variable_suffix"), specifying
        the mapping of group to numerical category
        for each observation. This can be used in the model definition to
        index the group membership of each observation.

        :param str variable_suffix: The suffix to append to the grouping variables in the dataframe.
        :param subsample_size: Passed to numpyro.plate argument of the same name.
        """
        columns_with_suffix = [
            col for col in self.data.columns if col.endswith(variable_suffix)
        ]
        if any(columns_with_suffix):
            msg = (
                f"The dataframe contains columns with the suffix {variable_suffix} already: \n{columns_with_suffix}\n"
                "Change the suffix to avoid conflicts by passing a new variable_suffix "
                "to _create_plates at instance construction."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.plate_dicts = {}
        for i, group in enumerate(self.group_variables):
            self.data[group] = self.data[group].astype("category")
            self.data[group + variable_suffix] = self.data[group].cat.codes
            logger.info(f"\nAdded new variable {group + variable_suffix} to self data.")
            coords = self._cats_to_dict(self.data[group])
            dim = -(i + 1)
            size = len(self.data[group + variable_suffix].unique())
            idx = self.data[
                group + variable_suffix
            ].values  # index of group in the data
            self.plate_dicts[group] = {}
            self.plate_dicts[group]["coords"] = coords
            self.plate_dicts[group]["dim"] = dim
            self.plate_dicts[group]["size"] = size
            self.plate_dicts[group]["idx"] = idx
            # create the actual plate object:
            self.plate_dicts[group]["plate"] = numpyro.plate(
                name=group, size=size, dim=dim, subsample_size=subsample_size
            )

    def model(self):
        raise NotImplementedError("You must overrwrite the default model method!")

    def generate_reparam_config(self) -> dict:
        logger.info(
            (
                "No reparameterization currently defined. "
                "If you expect reparam=True to have any effect you "
                "must overwrite this method.",
            )
        )
        return {}

    def generate_arviz_data(
        self,
        num_samples: int = 1000,
        dims: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        predictive_kwargs: Optional[dict] = None,
        from_numpyro_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Generate data in the arviz format

        This method allows you to generate an InferenceData
        object suitable for use with Arviz. The method is mostly
        a wrapper for Arviz' function from_numpryo.

        It will generate both posterior and prior predictive samples
        using the predict method, unless these have already been
        generated by the user calling `predict` on the self.data
        already. In the latter case, predictive_kwargs will be
        ignored.

        To improve the meaning of InferenceData further, this
        method tries to intelligently add information from the plates
        in plate_dict to the coordinates of the arviz data.
        When using plates in your model, you can also pass a dictionary
        to the `dims` argument. This dictionary should specify
        the correspondence between the plate name (same as the
        variable in the dataframe corresponding to the group) and
        the model node. Example below.

        :param num_samples: The number of samples to generate from
            the predictive distribution(s).
        :param dict[str] -> list[str] dims: A dictionary specifying the
            correspondence between data labels (coordinates) and
            inference nodes (model variables).
            Example: {"model_param_a": ["data_variable_g"]}.
        :param model_kwargs: Keyword arguments passed to the model.
        :param dict predictive_kwargs: Keyword arguments passed to
            Numpyro's Predictive class.
        :param from_numpyro_kwargs: Keyword arguments passed to arviz's
            from_numpyro function.
        """
        model_kwargs = model_kwargs or {}
        predictive_kwargs = predictive_kwargs or {}
        from_numpyro_kwargs = from_numpyro_kwargs or {}

        if self.posterior_predictive is None:
            # only compute if not already computed
            posterior_predictive = self.predict(
                prior=False,
                num_samples=num_samples,
                model_kwargs=model_kwargs,
                predictive_kwargs=predictive_kwargs,
            )
            self.posterior_predictive = posterior_predictive
        else:
            logger.info(
                (
                    "Found an existing set of posterior predictions "
                    "for the instance data. Using those. Any arguments you passed "
                    "to predictive_kwargs will be ignored."
                )
            )
            posterior_predictive = self.posterior_predictive

        if self.prior_predictive is None:
            # only compute if not already computed
            prior_predictive = self.predict(
                prior=True,
                num_samples=num_samples,
                model_kwargs=model_kwargs,
                predictive_kwargs=predictive_kwargs,
            )
            self.prior_predictive = prior_predictive
        else:
            prior_predictive = self.prior_predictive
            logger.info(
                (
                    "Found an existing set of prior predictions "
                    "for the instance data. Using those. Any arguments you passed "
                    "to predictive_kwargs will be ignored."
                )
            )

        if "coords" in from_numpyro_kwargs.keys():
            logger.warning(
                (
                    "You tried to pass coords to from_numpyro_kwargs,"
                    " but this should be handled internally by numpyro-oop. "
                    "Your option will be ignored."
                )
            )

        if self.plate_dicts is not None:
            coords = {}
            for plate in self.plate_dicts.keys():
                coords[plate] = list(self.plate_dicts[plate]["coords"].values())
        else:
            coords = None

        arviz_data = az.from_numpyro(
            self.mcmc,
            prior=prior_predictive,
            posterior_predictive=posterior_predictive,
            coords=coords,
            dims=dims,
            **from_numpyro_kwargs,
        )
        self.arviz_data = arviz_data
