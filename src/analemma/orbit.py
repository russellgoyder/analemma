"""
Implementation of formulae for calculating orbits
"""

import math
import numpy as np
from numpy import sin, cos, typing as npt
from dataclasses import dataclass, field
import datetime
from functools import lru_cache
from skyfield import almanac
from skyfield.api import load
from skyfield import searchlib as sf_search


pi = math.pi


@dataclass
class PlanetParameters:
    """
    Parameters defining a planet for sundial calculation purposes

    Parameters:
        N: Number of mean days in a year
        T_d: Number of seconds in a mean day
        rho: Angle between axes of the ellipse and the equinoxes / solstices
        alpha: Inclination of the earths axis of rotation
        a: Length of the planet's orbit's semi-major axis
        e: Eccentricity of the planet's orbit

    The following attributes are calculated given the above parameters.

    Attributes:
        T_y: Number of seconds in a mean year
        om_y: Mean angular speed of the earth's centre of mass in its orbit
        om_d: Angular speed of a point on the earth about the earth's centre of mass
        om_sd: Angular speed of a point on an earth that revolves once per siderial day
        T_sd: Number of seconds in a siderial day
        Om: Angular speed parameter used in the spinor orbit formalism ( = om_y / 2 * a )
    """

    N: float
    T_d: int
    rho: float
    alpha: float
    a: float
    e: float
    T_y: int = field(init=False)
    om_y: float = field(init=False)
    om_d: float = field(init=False)
    om_sd: float = field(init=False)
    T_sd: float = field(init=False)
    Om: float = field(init=False)

    def __post_init__(self):
        self.T_y = self.N * self.T_d
        self.om_y = 2 * pi / self.T_y
        self.om_d = 2 * pi / self.T_d
        self.om_sd = (self.N + 1) / self.N * self.om_d
        self.T_sd = self.N / (self.N + 1) * self.T_d
        self.Om = pi / self.T_y * self.a

    def clone_with_eccentricity(self, e: float):
        """
        Return a new instance with the same parameters but a different eccentricity
        """
        return PlanetParameters(
            N=self.N,
            T_d=self.T_d,
            rho=self.rho,
            alpha=self.alpha,
            a=self.a,
            e=e,
        )

    def daily_noons(self) -> np.array:
        """
        Daily time samples in seconds from noon at perihelion
        """
        return self.T_d * np.arange(int(self.N))

    def rotation_angle(self, t: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Angle of planetary rotation increases linearly with time, one complete revolution
        after one siderial day, and an offset of rho

        Parameters:
            t: Collection of time in seconds starting at perihelion at which to calculate the planet's angle of rotation

        Returns:
            The angle $\psi$ measuring planetary rotation at the input times
        """
        return np.mod(self.rho + self.om_sd * t, 2 * pi)

    def orbit_angle(self, t: npt.ArrayLike) -> npt.ArrayLike:
        """

        Parameters:
            t: Collection of time in seconds starting at perihelion at which to calculate the planet's angle of rotation
        """
        phi = orbital_angle(spinor_time(t))
        return np.mod(
            pi + self.rho + phi, 2 * pi
        )  # phi starts at perihelion, sigma starts at winter solstice

    def __hash__(self):
        # this is not as efficient as it could be but covers the
        # common case of reuse of the same object for a given planet
        return id(self)

    @classmethod
    def earth(cls):
        """
        Return a PlanetParameters instance representing Earth
        """
        return PlanetParameters(
            N=365.2422,
            T_d=24 * 3600,
            rho=12.25 / 180 * pi,
            alpha=23.5 / 180 * pi,
            a=149598000000,
            e=0.017,
        )


earth = PlanetParameters.earth()
"""
An instance of PlanetParameters representing Earth
"""


def _kepler_params(planet: PlanetParameters = earth):
    a = planet.a
    e = planet.e
    b = a * math.sqrt(1 - e**2)  # semi-minor axis
    A = math.sqrt((a + b) / 2)
    B = math.sqrt((a - b) / 2)
    return A, B, planet.Om, planet.T_y


def orbital_time(s: npt.ArrayLike, planet: PlanetParameters = earth):
    """
    Calculate orbital time given time parameter, t(s)

    Parameters:
        s: Spinor time parameter (note, called $\tau$ in the paper)
        planet: Planet whose orbit is being analyzed
        e: Optional override for the orbit's eccentricity
    """
    A, B, Om, T_y = _kepler_params(planet)
    return (A**2 + B**2) * s + A * B / Om * sin(2 * Om * s) + T_y / 2


_cache_max_size = 50


@lru_cache(maxsize=_cache_max_size)
def _finegrained_interp_points(planet: PlanetParameters):
    s_points = np.linspace(-pi / planet.Om / 2, pi / planet.Om / 2, 10_000)
    t_points = orbital_time(s_points, planet)
    return s_points, t_points


def spinor_time(t: npt.ArrayLike, planet: PlanetParameters = earth):
    """
    Invert t(s), the relationship of orbital time t with the parameter in the spinor
    treatment of the Kepler problem, s, to give s(t).
    """
    s_points, t_points = _finegrained_interp_points(planet)
    return np.interp(t, t_points, s_points)


def orbital_radius(s: npt.ArrayLike, planet: PlanetParameters = earth):
    """
    Calculate orbital radial coordinate given spinor time parameter, r(s)
    """
    A, B, Om, _ = _kepler_params(planet)
    return A**2 + B**2 + 2 * A * B * cos(2 * Om * s)


def orbital_angle(s: npt.ArrayLike, planet: PlanetParameters = earth):
    """
    Calculate orbital angular coordinate given time parameter, phi(s)
    """
    A, B, Om, _ = _kepler_params(planet)
    tanSigY = (A**2 - B**2) * sin(2 * Om * s)
    tanSigX = (A**2 + B**2) * cos(2 * Om * s) + 2 * A * B
    return np.arctan2(tanSigY, tanSigX) + pi


@lru_cache(maxsize=_cache_max_size)
def _skyfield_ephemeris():
    return load("de440s.bsp"), load.timescale()


def _skyfield_season_events(year: int):
    eph, ts = _skyfield_ephemeris()
    jan1 = ts.utc(year, 1, 1)
    dec31 = ts.utc(year, 12, 31)
    event_times, _ = almanac.find_discrete(jan1, dec31, almanac.seasons(eph))
    return event_times


@dataclass
class OrbitDateAndAngle:
    """
    Pairing of a date and the corresponding orbit angle
    """

    date: datetime.date
    sigma: float


def season_event_info(season_value: int, year: int) -> OrbitDateAndAngle:
    """
    Return the date and orbit angle for an equinox or solstice in a given year, identified by
    the season's value as per [analemma.geometry.Season][].
    """
    season_events = _skyfield_season_events(year)
    # S S A W (seasons)
    # 1 2 3 0 (Season enum)
    # 0 1 2 3 (Skyfield)
    skyfield_season_value = (season_value + 3) % 4
    season_event_angles = [pi / 2, 0, 3 * pi / 2, pi]
    return OrbitDateAndAngle(
        season_events[skyfield_season_value].utc_datetime().date(),
        season_event_angles[skyfield_season_value],
    )


def _earth_distance(time):
    eph, _ = _skyfield_ephemeris()
    earth = eph["earth"]
    sun = eph["sun"]
    e = earth.at(time)
    s = e.observe(sun)
    return s.distance().km


_earth_distance.step_days = 1


def _perihelion_date(year: int) -> datetime.date:
    _, ts = _skyfield_ephemeris()
    start_time = ts.utc(year, 1, 1)
    end_time = ts.utc(year + 1, 1, 1)
    perihelion = sf_search.find_minima(start_time, end_time, _earth_distance)
    return perihelion[0][0].utc_datetime().date()


def orbit_day_to_date(orbit_day: int, year: int = None) -> datetime.date:
    """
    Convert from the number of days since perihelion to the date
    """
    if not year:
        year = datetime.date.today().year
    return _perihelion_date(year) + datetime.timedelta(days=int(orbit_day))


def orbit_date_to_day(the_date: datetime.date, year: int = None) -> int:
    """
    Convert from the date to the number of days since perihelion
    """
    if not year:
        year = datetime.date.today().year
    return (the_date - _perihelion_date(year)).days
