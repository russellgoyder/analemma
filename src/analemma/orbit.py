"""
Implementation of formulae for calculating orbits
"""

import math, numpy as np
from numpy import sin, cos
from dataclasses import dataclass, field


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


def _kepler_params(planet: PlanetParameters = earth, e : float = None):
    a = planet.a
    if not e:
        e = planet.e
    b = a * math.sqrt(1 - e**2)  # semi-minor axis
    A = math.sqrt((a + b) / 2)
    B = math.sqrt((a - b) / 2)
    return A, B, planet.Om, planet.T_y


def orbital_time(s : np.array, planet: PlanetParameters = earth, e: float = None):
    """
    Calculate orbital time given time parameter, t(s)

    Parameters:
        s: Spinor time parameter (note, called $\tau$ in the paper)
        planet: Planet whose orbit is being analyzed
        e: Optional override for the orbit's eccentricity
    """
    A, B, Om, T_y = _kepler_params(planet, e)
    return (A**2 + B**2) * s + A * B / Om * sin(2 * Om * s) + T_y / 2


_s_finegrained = np.linspace(-pi / earth.Om / 2, pi / earth.Om / 2, 10_000)
"""
1-d grid used when inverting the relationship between orbital and spinor time
"""

def _key(e: float) -> int:
    """
    The first four significant figures of the given number
    """
    return int(10_000 * e)


_t_finegrained = {_key(earth.e): orbital_time(_s_finegrained)}
"""
Cache of interpolation data used when inverting the relationship between orbital and spinor time 
"""

def spinor_time(t : np.array, planet: PlanetParameters = earth, e : float = None):
    """
    Invert t(s), the relationship of orbital time t with the parameter in the spinor
    treatment of the Kepler problem, s, to give s(t).

    Keep a cache of interpolants, one per eccentricity.
    """
    if not e:
        e = planet.e
    k = _key(e)
    if k not in _t_finegrained.keys():
        _t_finegrained[k] = orbital_time(_s_finegrained, planet, e)
    return np.interp(t, _t_finegrained[k], _s_finegrained)


def orbital_radius(s : np.array, planet: PlanetParameters = earth, e: float = None):
    """
    Calculate orbital radial coordinate given spinor time parameter, r(s)
    """
    A, B, Om, _ = _kepler_params(planet, e)
    return A**2 + B**2 + 2 * A * B * cos(2 * Om * s)


def orbital_angle(s, planet: PlanetParameters = earth, e=None):
    """
    Calculate orbital angular coordinate given time parameter, phi(s)
    """
    A, B, Om, _ = _kepler_params(planet, e)
    tanSigY = (A**2 - B**2) * sin(2 * Om * s)
    tanSigX = (A**2 + B**2) * cos(2 * Om * s) + 2 * A * B
    return np.arctan2(tanSigY, tanSigX) + pi
