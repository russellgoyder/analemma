"""
Implementation of formulae for calculating sundial geometry
"""

import numpy as np
from numpy import sin, cos, typing as npt
from typing import Tuple


def hour_angle_terms(
    alpha: npt.ArrayLike,
    sigma: npt.ArrayLike,
    psi: npt.ArrayLike,
    iota_minus_theta: npt.ArrayLike = np.nan,
) -> Tuple[np.array, np.array]:
    """
    Generalized hour angle of the sun measured as the angle between the gnomon and a sun ray

    Return the numerator and denominator in the tangent of the angle. Each contains a factor
    of sin(Xi) which cancels in the ratio.

    When iota_minus_theta is zero, this reduces to the common definition of hour angle

    Parameters:
        alpha: Tilt of Earth's axis of rotation from normal to the plane of its orbit
        sigma: Angle between Earth-Sun vector and the same at summer solstice
        psi: Measuring the rotation of the Earth
        iota_minus_theta: Angle of gnomon in the meridian plane relative to latitude
    """

    if np.isnan(iota_minus_theta):
        iota_minus_theta = 0.0

    sinXi_sin_mu = sin(psi) * cos(sigma) * cos(alpha) - cos(psi) * sin(sigma)

    term1 = cos(psi) * cos(sigma) * cos(alpha) + sin(psi) * sin(sigma)
    sinXi_cos_mu = term1 * cos(iota_minus_theta) - cos(sigma) * sin(alpha) * sin(
        iota_minus_theta
    )

    return (sinXi_sin_mu, sinXi_cos_mu)


def hour_angle(
    alpha: npt.ArrayLike,
    sigma: npt.ArrayLike,
    psi: npt.ArrayLike,
    iota_minus_theta: npt.ArrayLike = np.nan,
) -> np.array:
    "Evaluate the inverse tangent of the sun's hour angle"

    sinXi_sin_mu, sinXi_cos_mu = hour_angle_terms(alpha, sigma, psi, iota_minus_theta)
    return np.arctan2(sinXi_sin_mu, sinXi_cos_mu)


def shadow_denom(
    alpha: npt.ArrayLike,
    sigma: npt.ArrayLike,
    psi: npt.ArrayLike,
    theta: npt.ArrayLike,
    i: npt.ArrayLike,
    d: npt.ArrayLike,
) -> np.array:
    "The denominator in the shadow coordinate expressions"

    sinXi_sin_mu_s, sinXi_cos_mu_s = hour_angle_terms(
        alpha, sigma, psi
    )  # calc with gnomon as a style
    return (
        sinXi_cos_mu_s * (sin(i) * cos(d) * cos(theta) - sin(theta) * cos(i))
        - sinXi_sin_mu_s * sin(d) * sin(i)
    ) + (sin(i) * sin(theta) * cos(d) + cos(i) * cos(theta)) * sin(alpha) * cos(sigma)


def shadow_coords_xy(
    alpha: npt.ArrayLike,
    sigma: npt.ArrayLike,
    psi: npt.ArrayLike,
    iota: npt.ArrayLike,
    theta: npt.ArrayLike,
    i: npt.ArrayLike,
    d: npt.ArrayLike,
) -> np.array:
    """
    Calculate the x and y coordinates of the tip of the shadow in the frame embedded in the dial face

    alpha, sigma, psi, iota and theta are as defined in sd_hour_angle_terms. The angles i and d
    define the orientation of the dial face.
    """

    sinXi_sin_mu, sinXi_cos_mu = hour_angle_terms(alpha, sigma, psi, iota - theta)

    D_denom = shadow_denom(alpha, sigma, psi, theta, i, d)

    x = (-sin(d) * sinXi_sin_mu * cos(iota) + cos(d) * sinXi_cos_mu) / D_denom
    y = (
        -(
            sin(d) * cos(i) * sinXi_cos_mu
            + sin(i) * sin(iota) * sinXi_sin_mu
            + sinXi_sin_mu * cos(d) * cos(i) * cos(iota)
        )
        / D_denom
    )

    return (x, y)
