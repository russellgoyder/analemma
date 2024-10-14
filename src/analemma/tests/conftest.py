import pytest
import numpy as np
from analemma import geometry as geom, orbit


@pytest.fixture
def earth():
    return orbit.PlanetParameters.earth()


@pytest.fixture
def camdial():
    return geom.DialParameters(
        theta=37.5 / 180 * np.pi, iota=37.5 / 180 * np.pi, i=0, d=0
    )  # Analemmatic dial in Cambridge, UK
