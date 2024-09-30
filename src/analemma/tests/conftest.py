import pytest
import numpy as np
from analemma import orbit
from analemma import plot as ap


@pytest.fixture
def earth():
    return orbit.PlanetParameters.earth()


@pytest.fixture
def camdial():
    return ap.DialParameters(
        theta=37.5 / 180 * np.pi, iota=37.5 / 180 * np.pi, i=0, d=0
    )  # Analemmatic dial in Cambridge, UK
