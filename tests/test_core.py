import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

from numpyro_oop.core import BaseNumpyroModel


class DummyModel(BaseNumpyroModel):
    def model(self, data=None):
        if data is None:
            data = self.data
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
        M = b * data["x"].values
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = a + M
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data["y"].values)


@pytest.fixture
def dummy_data():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 1, 2]})


def test_init(dummy_data):
    model_with_data = DummyModel(seed=42, data=dummy_data)
    pd.testing.assert_frame_equal(model_with_data.data, dummy_data)

    model_without_data = DummyModel(seed=42)
    assert model_without_data.data is None


# Check if the model raises an error or behaves unexpectedly when no data is provided
def test_model_behavior_without_data():
    model = DummyModel(seed=42)
    with pytest.raises(Exception):
        model.sample()
    with pytest.raises(Exception):
        model.predict()


# Check model sampling runs
def test_model_sample(dummy_data):
    model = DummyModel(seed=42, data=dummy_data)
    model.sample(num_samples=500, num_warmup=500, num_chains=2)
    # how to test the values? Not deterministic.
    # at least will throw an error if it doesn't run.
