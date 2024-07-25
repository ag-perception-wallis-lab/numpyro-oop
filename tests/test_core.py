import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.primitives
import pandas as pd
import pytest

from numpyro_oop.core import BaseNumpyroModel


class DummyModel(BaseNumpyroModel):
    def model(self, data=None):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
        M = b * data["x"].values
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = numpyro.deterministic("mu", a + M)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data["y"].values)


class DummyModelHierarchical(BaseNumpyroModel):
    def model(self, data=None):
        idx = self.plate_dicts["a_categorical"]["idx"]

        # a random intercept model:
        a_mu = numpyro.sample("a_mu", dist.Normal(0.0, 1.0))
        a_sigma = numpyro.sample("a_sigma", dist.LogNormal(0.0, 0.5))

        with self.plate_dicts["a_categorical"]["plate"]:
            a = numpyro.sample("a", dist.Normal(a_mu, a_sigma))

        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
        M = b * data["x"].values
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = numpyro.deterministic("mu", a[idx] + M)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data["y"].values)


@pytest.fixture(scope="session")
def dummy_data():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            "y": [0.0, 0.8, 1.2, 2.5, 3.2, 6.4],
            "a_categorical": ["A", "B", "A", "B", "A", "B"],
        }
    )


@pytest.fixture(scope="session")
def dummy_fitted(dummy_data):
    m1 = DummyModel(seed=42, data=dummy_data)
    m1.sample(num_samples=500, num_warmup=500, num_chains=2)
    return m1


@pytest.fixture(scope="session")
def dummy_fitted_hierarhical(dummy_data):
    m1 = DummyModelHierarchical(
        seed=42, data=dummy_data, group_variables="a_categorical"
    )
    m1.sample(num_samples=1000, num_warmup=1000, num_chains=3)
    return m1


def test_init(dummy_data):
    model_with_data = DummyModel(seed=42, data=dummy_data)
    pd.testing.assert_frame_equal(model_with_data.data, dummy_data)

    model_without_data = DummyModel(seed=42)
    assert model_without_data.data is None


def test_cats_to_dict():
    x = pd.Series(["A", "B", "C", "D"]).astype("category")
    assert DummyModel._cats_to_dict(x) == {0: "A", 1: "B", 2: "C", 3: "D"}

    x = pd.Series(["A", "B", "C", "D"])  # not a category
    with pytest.raises(Exception):
        DummyModel._cats_to_dict(x)


def test_init_with_group(dummy_data):
    m1 = DummyModel(seed=42, data=dummy_data, group_variables="a_categorical")
    assert "a_categorical_id" in m1.data.columns
    assert all(m1.data.loc[m1.data["a_categorical"] == "A", "a_categorical_id"] == 0)
    assert all(m1.data.loc[m1.data["a_categorical"] == "B", "a_categorical_id"] == 1)
    assert m1.group_variables == ["a_categorical"]
    assert type(m1.plate_dicts) == dict
    assert m1.plate_dicts["a_categorical"]["coords"] == {0: "A", 1: "B"}
    assert m1.plate_dicts["a_categorical"]["dim"] == -1
    assert m1.plate_dicts["a_categorical"]["size"] == 2
    jnp.array_equal(
        m1.plate_dicts["a_categorical"]["idx"], jnp.array([0, 1, 0, 1, 0, 1])
    )
    assert type(m1.plate_dicts["a_categorical"]["plate"]) == numpyro.primitives.plate


# Check if the model raises an error or behaves unexpectedly when no data is provided
def test_model_behavior_without_data():
    m1 = DummyModel(seed=42)
    with pytest.raises(Exception):
        m1.sample()
    with pytest.raises(Exception):
        m1.predict()


# Check model sampling runs
def test_model_samples(dummy_fitted):
    a_mean = dummy_fitted.posterior_samples["a"].mean()
    b_mean = dummy_fitted.posterior_samples["b"].mean()
    sigma_mean = dummy_fitted.posterior_samples["sigma"].mean()
    sigma_sd = dummy_fitted.posterior_samples["sigma"].std()

    # test against the "checked" reference values:
    assert jnp.allclose(-0.595, a_mean, atol=0.01), f"Mean of a is incorrect: {a_mean}"
    assert jnp.allclose(0.710, b_mean, atol=0.01), f"Mean of b is incorrect: {b_mean}"
    assert jnp.allclose(
        0.354, sigma_mean, atol=0.01
    ), f"Mean of sigma is incorrect: {sigma_mean}"
    assert jnp.allclose(
        0.182, sigma_sd, atol=0.01
    ), f"Sigma std is incorrect: {sigma_sd}"


def test_posterior_predictive(dummy_fitted):
    yhat = dummy_fitted.predict()["mu"]
    assert yhat.shape == (1000, 6)
    yhat_mean = yhat.mean()
    assert jnp.allclose(
        2.366, yhat_mean, atol=0.01
    ), f"Yhat mean is incorrect: {yhat_mean}"


# Check hierarchical sampling runs
def test_model_samples_hierarchical(dummy_fitted_hierarhical):
    assert dummy_fitted_hierarhical.posterior_samples["a_mu"].shape == (3000,)
    assert dummy_fitted_hierarhical.posterior_samples["a"].shape == (3000, 2)
    a_mu_mean = dummy_fitted_hierarhical.posterior_samples["a_mu"].mean()
    a_mean = dummy_fitted_hierarhical.posterior_samples["a"].mean()
    b_mean = dummy_fitted_hierarhical.posterior_samples["b"].mean()
    sigma_mean = dummy_fitted_hierarhical.posterior_samples["sigma"].mean()
    sigma_sd = dummy_fitted_hierarhical.posterior_samples["sigma"].std()

    # test against the "checked" reference values:
    assert jnp.allclose(
        -0.378, a_mu_mean, atol=0.01
    ), f"Mean of a_mu is incorrect: {a_mu_mean}"
    assert jnp.allclose(-0.534, a_mean, atol=0.01), f"Mean of a is incorrect: {a_mean}"
    assert jnp.allclose(0.695, b_mean, atol=0.01), f"Mean of b is incorrect: {b_mean}"
    assert jnp.allclose(
        0.412, sigma_mean, atol=0.01
    ), f"Mean of sigma is incorrect: {sigma_mean}"
    assert jnp.allclose(
        0.221, sigma_sd, atol=0.01
    ), f"Sigma std is incorrect: {sigma_sd}"
