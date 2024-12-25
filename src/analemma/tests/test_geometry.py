import pytest
import numpy as np
from sympy import utilities as util
from analemma import geometry as geom
from analemma.algebra import frame, result


pi = np.pi
hour_angle_args = ["alpha", "sigma", "psi", "iota", "theta"]


@pytest.mark.parametrize(
    ",".join(hour_angle_args),
    [
        (0, 0, 0, 0, 0),
        (23.5 / 180 * pi, 0, 0, 0, 0),  # add Earth axis tilt
        (pi / 2, 0, 0, 0, 0),  # 90 degree tilt
        (
            23.5 / 180 * pi,
            pi / 2,
            0.1,
            0,
            0,
        ),  # September equinox (non-zero psi to avoid tan singularity)
        (23.5 / 180 * pi, pi, 0, 0, 0),  # December solstice
        (
            23.5 / 180 * pi,
            3 * pi / 2,
            0.1,
            0,
            0,
        ),  #  March equinox (non-zero psi to avoid tan singularity)
        (23.5 / 180 * pi, 2 * pi, 0, 0, 0),  # June solstice
        (23.5 / 180 * pi, pi / 7, 71 / 180 * pi, 0, 0),  # arbitrary psi
        (23.5 / 180 * pi, pi / 7, 71 / 180 * pi, 0, pi / 2),  # equator
        (23.5 / 180 * pi, pi / 7, 71 / 180 * pi, 0, pi),  # South pole
        (
            23.5 / 180 * pi,
            pi / 7,
            71 / 180 * pi,
            0,
            68 / 180 * pi,
        ),  # arbitrary latitude
        (
            23.5 / 180 * pi,
            pi / 7,
            71 / 180 * pi,
            pi / 6,
            68 / 180 * pi,
        ),  # gnomon with 30 degree tilt
        (
            23.5 / 180 * pi,
            pi / 7,
            71 / 180 * pi,
            79 / 180 * pi,
            68 / 180 * pi,
        ),  # gnomon almost touching dial face
    ],
)
def test_hour_angle(alpha, sigma, psi, iota, theta):
    """
    Tests for the implementation of the hour angle calculation

    See [analemma.geometry.hour_angle_terms][] and [analemma.geometry.hour_angle][]
    """

    # alegbraic result
    g = frame.gnomon("e", zero_decl=True)
    s = frame.sunray()
    S = result.shadow_bivector(s, g)
    M = frame.meridian_plane()
    sinXi_sin_mu, sinXi_cos_mu = result.hour_angle_sincos(S, M)

    # convert to numerical functions
    sin_term = util.lambdify(hour_angle_args, sinXi_sin_mu)
    cos_term = util.lambdify(hour_angle_args, sinXi_cos_mu)

    # evaluate and ensure consistent with implementation in geometry module
    sin_val, cos_val = geom.hour_angle_terms(alpha, sigma, psi, iota - theta)
    assert sin_term(alpha, sigma, psi, iota, theta) == pytest.approx(sin_val)
    assert cos_term(alpha, sigma, psi, iota, theta) == pytest.approx(cos_val)

    # ensure depends only on difference between iota and theta
    arbitrary_shift = pi / 13
    assert sin_term(alpha, sigma, psi, iota, theta) == pytest.approx(
        sin_term(alpha, sigma, psi, iota + arbitrary_shift, theta + arbitrary_shift)
    )
    assert cos_term(alpha, sigma, psi, iota, theta) == pytest.approx(
        cos_term(alpha, sigma, psi, iota + arbitrary_shift, theta + arbitrary_shift)
    )

    # test some periodicity
    assert geom.hour_angle_terms(alpha, sigma, psi, iota - theta) == pytest.approx(
        geom.hour_angle_terms(
            alpha + 2 * pi, sigma + 4 * pi, psi + 10 * pi, iota - theta + 50 * pi
        )
    )

    # combine the terms
    assert np.tan(geom.hour_angle(alpha, sigma, psi, iota - theta)) == pytest.approx(
        sin_val / cos_val
    )


@pytest.mark.parametrize(("latitude"), [0, 40, 50, 90, -10, -90, -0])
def test_common_dial_types_basic_construction(latitude: float):
    eq = geom.DialParameters.equatorial(latitude=latitude)
    theta = (90 - latitude) / 180 * pi
    assert eq.theta == pytest.approx(theta)
    assert eq.theta == pytest.approx(eq.iota) == pytest.approx(eq.i)
    assert pytest.approx(eq.d) == pytest.approx(0)

    hor = geom.DialParameters.horizontal(latitude=latitude)
    theta = (90 - latitude) / 180 * pi
    assert hor.theta == pytest.approx(theta)
    assert hor.theta == pytest.approx(hor.iota)

    vert = geom.DialParameters.vertical(latitude=latitude)
    theta = (90 - latitude) / 180 * pi
    assert vert.theta == pytest.approx(theta)
    assert vert.theta == pytest.approx(eq.iota)
    assert vert.i == pytest.approx(pi / 2)
    decl = pi if latitude >= 0 else 0
    assert vert.d == pytest.approx(decl)
