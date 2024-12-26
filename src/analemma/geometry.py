"""
Numerical functionality for calculate sundial geometry and analemma projections
"""

import numpy as np
from numpy import sin, cos, typing as npt
from scipy import optimize as sci_opt
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from analemma import orbit


pi = np.pi


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


@dataclass
class DialParameters:
    """
    Parameters defining a sundial

    Parameters:
        theta: $90^\\circ - \\theta$ is the latitude of the sundial
        iota: Inclination of the gnomon
        i: Inclination of the dial face
        d: Declination of the dial face
        x_length: Width of the dial face in gnomon lengths
        y_length: Length of the dial face in gnomon lengths
    """

    theta: float
    iota: float
    i: float
    d: float
    x_length: float = 10  # gnomon has unit length
    y_length: float = 10

    def trim_coords(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        """
        Set points falling outside the dial face to nan so they don't show up when plotted

        Parameters:
            x: x-values of a set of points in 2-d
            y: y-values of a set of points in 2-d

        Returns:
            The input arrays of x and y coordinates with those points falling outside the face of the sundial set to NaN
        """

        def dial_trim(vec, dial_length):
            return np.array(
                [coord if np.abs(coord) < dial_length else np.nan for coord in vec]
            )

        return dial_trim(x, self.x_length), dial_trim(y, self.y_length)

    def within_face(self, x: np.array, y: np.array) -> np.array:
        """
        Determine which of a collection of points falls within the face of a sundial

        Parameters:
            x: x-values of a set of points in 2-d
            y: y-values of a set of points in 2-d

        Returns:
            An array of boolean values indicating whether each input point falls within the face of the sundial
        """

        def _within_dialface(vec, dial_length):
            return np.array(
                [True if np.abs(coord) < dial_length else False for coord in vec]
            )

        return _within_dialface(x, self.x_length) & _within_dialface(y, self.y_length)

    @classmethod
    def equatorial(cls, latitude: float) -> "DialParameters":
        """
        An equatorial dial's face is parallel to the plane of the equator, and the style
        is therefore perpendicular to both.
        """
        theta = (90 - latitude) / 180 * pi
        return cls(theta=theta, iota=theta, i=theta, d=0)

    @classmethod
    def horizontal(cls, latitude: float) -> "DialParameters":
        """
        A horizontal dial's face is parallel to the ground, with a style aligned
        with the planet's axis of rotation.
        """
        theta = (90 - latitude) / 180 * pi
        return cls(theta=theta, iota=theta, i=0, d=0)

    @classmethod
    def vertical(cls, latitude: float) -> "DialParameters":
        """
        A vertical dial's face is perpendicular to the ground, and typically faces either
        south in the northern hemisphere and north in the southern hemisphere. Instances of
        this class represent a south-facing dial on the equator and in northern latitudes, and
        a noth-facing dial below the equator (in southern latitudes). Its gnomon
        is a style, aligned with the planet's axis of rotation.
        """
        d = pi if latitude >= 0 else 0
        theta = (90 - latitude) / 180 * pi
        return cls(theta=theta, iota=theta, i=pi / 2, d=d)


def sin_sunray_dialface_angle(
    t: np.array, planet: orbit.PlanetParameters, dial: DialParameters
):
    """
    Sine of the angle between the sun ray and the dial face
    """

    alpha = planet.alpha
    theta = dial.theta
    i = dial.i
    d = dial.d

    psi = planet.rotation_angle(t)
    sigma = planet.orbit_angle(t)

    # -(G^(-s))|I where G is the dial face, s the sunray and I the pseudoscalar
    val = (
        -sin(alpha) * sin(i) * sin(theta) * cos(d) * cos(sigma)
        - sin(alpha) * cos(i) * cos(sigma) * cos(theta)
        + sin(d) * sin(i) * sin(psi) * cos(alpha) * cos(sigma)
        - sin(d) * sin(i) * sin(sigma) * cos(psi)
        - sin(i) * sin(psi) * sin(sigma) * cos(d) * cos(theta)
        - sin(i) * cos(alpha) * cos(d) * cos(psi) * cos(sigma) * cos(theta)
        + sin(psi) * sin(sigma) * sin(theta) * cos(i)
        + sin(theta) * cos(alpha) * cos(i) * cos(psi) * cos(sigma)
    )
    return -val  # because (-s)


@dataclass
class _SunTime:
    """
    Time in seconds since perihelion

    Perihelion occurs close to noon (when f1 is parallel to e1)
    """

    absolute_seconds: float
    days_since_perihelion: int

    @property
    def hours_from_midnight(self):
        "Convert absolute time in seconds to hours since midnight"
        return self.absolute_seconds / 3600 - (self.days_since_perihelion - 0.5) * 24


class SunTimes:
    """
    Time of key events in the journey of the sun across the sky

    Time zero is at perihelion. Note that all such key events are defined relatiev to the dial, not the ground.

    Attributes:
        sunrise: The first time in the given day that the sun ray is parallel to the dial face
        noon: The time after sunrise and before sunset when the angle bewteen sun ray and dial face is largest
        sunset: The second time in the given day that the sun ray is parallel to the dial face
        days_since_perihelion: Integer defining the day
    """

    def __init__(
        self,
        sunrise: _SunTime,
        noon: _SunTime,
        sunset: _SunTime,
        days_since_perihelion: int = 0,
    ):
        self.sunrise = _SunTime(sunrise, days_since_perihelion)
        self.noon = _SunTime(noon, days_since_perihelion)
        self.sunset = _SunTime(sunset, days_since_perihelion)
        self.days_since_perihelion = days_since_perihelion

    def sample_times_for_one_day(self, res: int = 1000):
        """
        Generate an array of times suitable for sampling the progress of the sun across the sky
        """
        # time zero is at perihelion close to noon so start 12 hours back to capture sun rise and set in order
        raw_times = np.linspace(
            (self.days_since_perihelion - 0.5) * 24 * 3600,
            (self.days_since_perihelion + 0.5) * 24 * 3600,
            res,
        )
        return np.array(
            [_SunTime(raw_time, self.days_since_perihelion) for raw_time in raw_times]
        )

    def __str__(self):
        return f"sunrise: {self.sunrise.hours_from_midnight}, noon: {self.noon.hours_from_midnight}, sunset: {self.sunset.hours_from_midnight}"


def find_sun_rise_noon_set_relative_to_dial_face(
    days_since_perihelion: int, planet: orbit.PlanetParameters, dial: DialParameters
) -> SunTimes:
    """
    Find sunrise, noon and sunset (relative to the dial face)

    Note these because times are all relative to the dial face they do not match the common notions except for an analematic dial

    Parameters:
        days_since_perihelion: Integer defining the day
        planet: Parameters of the planet
        dial: Parameters of the sundial
    """

    # time when sunray meets dial face at pi/2 (90 degrees):
    t_noon_guess = (days_since_perihelion * 24) * 3600
    t_noon_result = sci_opt.minimize(
        lambda t: -sin_sunray_dialface_angle(t, planet, dial),
        x0=t_noon_guess,
        method="L-BFGS-B",
        tol=1.0e-8,
    )
    if not t_noon_result.success:
        raise Exception(
            f"Unable to find noon with days_since_perihelion = {days_since_perihelion} and initial guess of {t_noon_guess}. Optimization result: {t_noon_result}"
        )

    t_noon = t_noon_result.x[0]

    # times when sunray is parallel to dial face
    t_sunrise = sci_opt.brentq(
        lambda t: sin_sunray_dialface_angle(t, planet, dial), t_noon - 12 * 3600, t_noon
    )
    t_sunset = sci_opt.brentq(
        lambda t: sin_sunray_dialface_angle(t, planet, dial), t_noon, t_noon + 12 * 3600
    )

    return SunTimes(t_sunrise, t_noon, t_sunset, days_since_perihelion)


def sunray_dialface_angle(
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    orbit_day: int,
    hour_offset: float = 0.0,
):
    time = float(planet.T_d * orbit_day)
    time += hour_offset * 3600
    return sin_sunray_dialface_angle(time, planet, dial)


def sunray_dialface_angle_over_one_year(
    planet: orbit.PlanetParameters, dial: DialParameters, hour_offset: float = 0.0
):
    """
    Calculate daily the time since perihelion in seconds and the corresponding sin(sunray-dialface angle)
    """
    times = np.array([float(t) for t in planet.T_d * np.arange(0, 365)])
    times += hour_offset * 3600
    sines = np.array(sin_sunray_dialface_angle(times, planet, dial))
    return (times, sines)


def find_daytime_offsets(planet: orbit.PlanetParameters, dial: DialParameters):
    """
    Find the range of hours for which shadows are cast on the given dial

    The hours are returned as integer offsets from noon
    """

    tol = 0.01
    daytime_offsets = []
    for hour_offset in np.arange(-12, 12):
        _, sines = sunray_dialface_angle_over_one_year(planet, dial, hour_offset)
        if np.any(sines > tol):
            daytime_offsets.append(hour_offset)
    return daytime_offsets


class Season(Enum):
    """
    Enum to encode the four seasons of the year as seen temperate latitudes in the northern hemisphere.

    Start in Winter because orbit time starts at perihelion
    """

    Winter = 0
    Spring = 1
    Summer = 2
    Autumn = 3


def calc_raw_analemma_points(
    t: np.array, planet: orbit.PlanetParameters, dial: DialParameters
) -> Tuple[np.array, np.array, np.array]:
    """
    Calculate a collection of x and y coordinates of points on the projection of an analemma on a sundial

    For each given time, for the given sundial on the given planet, calculate the corresponding
    coordinate of the projection of the analemma onto the face of the dial. This corresponds to
    the tip of the shadow cast by the dial's gnomon.

    In addition, return an array of boolean values indicating whether each point falls within the face
    of the dial. See also [analemma.geometry.DialParameters.within_face][]

    Parameters:
        t: Array of times in seconds since perihelion
        planet: Parameters of the planet
        dial: Parameters of the sundial

    Returns:
        A 3-tuple `(x, y, w)` where `x` and `y` are arrays of coordinates and `w` is an array of Boolean values
    """
    psis = planet.rotation_angle(t)
    sigmas = planet.orbit_angle(t)
    x, y = shadow_coords_xy(
        planet.alpha, sigmas, psis, dial.iota, dial.theta, dial.i, dial.d
    )
    # when i == d == 0, the x axis (m1) points South and the y axis (m2) points East
    # more natural to rotate clockwise by pi/2 so that x points East and y points North
    xx = y
    yy = -x
    return xx, yy, dial.within_face(xx, yy)


def calc_analemma_points(
    t: np.array, planet: orbit.PlanetParameters, dial: DialParameters, trim: bool = True
) -> Tuple[np.array, np.array]:
    """
    Calculate a collection of x and y coordinates of points on the projection of an analemma on a sundial

    For each given time, for the given sundial on the given planet, calculate the corresponding
    coordinate of the projection of the analemma onto the face of the dial. This corresponds to
    the tip of the shadow cast by the dial's gnomon.

    In each array, if `trim` is `True`, the coordinate values will be set to NaN if the point does
    not fall within the face of the sundial.

    Parameters:
        t: Array of times in seconds since perihelion
        planet: Parameters of the planet
        dial: Parameters of the sundial

    Returns:
        A 2-tuple `(x, y)` where `x` and `y` are arrays of coordinates
    """
    x, y, _ = calc_raw_analemma_points(t, planet, dial)
    return dial.trim_coords(x, y) if trim else (x, y)
