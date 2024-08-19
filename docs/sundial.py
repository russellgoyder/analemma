"""
Implementation of formulae for calculating sundial geometry
"""

import math, numpy as np
from numpy import sin, cos
import dataclasses

def hour_angle_terms(alpha, sigma, psi, iota_minus_theta = np.nan):
    """
    Generalized hour angle of the sun measured as the angle between the gnomon and a sun ray

    Return the numerator and denominator in the tangent of the angle. Each contains a factor
    of sin(Xi) which cancels in the ratio.

    When iota_minus_theta is zero, this reduces to the common definition of hour angle

    :param alpha Tilt of Earth's axis of rotation from normal to the plane of its orbit
    :param sigma Angle between Earth-Sun vector and the same at summer solstice
    :param psi Measuring the rotation of the Earth
    :param iota_minus_theta Angle of gnomon in the meridian plane relative to latitude
    """

    if np.isnan(iota_minus_theta):
        iota_minus_theta = 0.0    

    sinXi_sin_mu = sin(psi)*cos(sigma)*cos(alpha) - cos(psi)*sin(sigma)

    term1  = cos(psi)*cos(sigma)*cos(alpha) + sin(psi)*sin(sigma)
    sinXi_cos_mu = term1*cos(iota_minus_theta) - cos(sigma)*sin(alpha)*sin(iota_minus_theta)
    
    return (sinXi_sin_mu, sinXi_cos_mu)

def hour_angle(alpha, sigma, psi, iota_minus_theta = np.nan):
    "Evaluate the inverse tangent of the sun's hour angle"

    sinXi_sin_mu, sinXi_cos_mu = hour_angle_terms(alpha, sigma, psi, iota_minus_theta)
    return np.arctan2(sinXi_sin_mu, sinXi_cos_mu)

def _D(alpha, sigma, psi, theta, i, d):
    "The denominator in the shadow coordinate expressions"

    sinXi_sin_mu_s, sinXi_cos_mu_s = hour_angle_terms(alpha, sigma, psi) # calc with gnomon as a style
    return (sinXi_cos_mu_s*(sin(i)*cos(d)*cos(theta) - sin(theta)*cos(i)) - sinXi_sin_mu_s*sin(d)*sin(i)) \
        + (sin(i)*sin(theta)*cos(d) + cos(i)*cos(theta))*sin(alpha)*cos(sigma)

def shadow_coords_xy(alpha, sigma, psi, iota, theta, i, d):
    """
    Calculate the x and y coordinates of the tip of the shadow in the frame embedded in the dial face

    alpha, sigma, psi, iota and theta are as defined in sd_hour_angle_terms. The angles i and d
    define the orientation of the dial face.
    """
    
    sinXi_sin_mu, sinXi_cos_mu = hour_angle_terms(alpha, sigma, psi, iota - theta)
    
    D_denom = _D(alpha, sigma, psi, theta, i, d)
    
    x = (-sin(d)*sinXi_sin_mu*cos(iota) + cos(d)*sinXi_cos_mu) / D_denom
    y = -(sin(d)*cos(i)*sinXi_cos_mu + sin(i)*sin(iota)*sinXi_sin_mu + sinXi_sin_mu*cos(d)*cos(i)*cos(iota)) / D_denom

    return (x, y)


pi = math.pi

@dataclasses.dataclass
class EotConstants:
    N : float = 365.2422 # number of mean days in a year
    T_d : int = 24 * 3600 # number of seconds in a mean day
    T_y : int = N * T_d # number of seconds in a mean year    
    om_y : float = 2*pi / T_y # mean angular speed of the earth's centre of mass in its orbit
    om_d : float = 2*pi / T_d # angular speed of a point on the earth about the earth's centre of mass
    om_sd : float = (N+1) / N * om_d # angular speed of a point on an earth that revolves once per siderial day
    T_sd : float = N / (N+1) * T_d # number of seconds in a siderial day
    rho : float = 12.25 / 180 * pi # angle between axes of the ellipse and the equinoxes / solstices
    alpha : float = 23.5 / 180 * pi # inclination of the earths axis of rotation
    a : float = 149598000000 # earth-sun orbit semi-major axes length in metres
    e : float = 0.017 # eccentricity of the earth-sun orbit
    Om : float = pi / T_y * a # angular speed parameter used in spinor orbit formalism ( = om_y / 2 * a )

def _kepler_params(C = EotConstants(), e = None):
    a = C.a
    if not e:
        e = C.e
    b = a*math.sqrt(1-e**2) # semi-minor axis
    A = math.sqrt((a+b)/2)
    B = math.sqrt((a-b)/2)
    return A, B, C.Om, C.T_y
    
def orbital_time(s, C = EotConstants(), e = None):
    "Calculate orbital time given time parameter, t(s)"
    A, B, Om, T_y = _kepler_params(C, e)
    return (A**2+B**2)*s + A*B/Om*sin(2*Om*s) + T_y/2

C = EotConstants()
_s_finegrained = np.linspace(-pi/C.Om/2, pi/C.Om/2, 10_000)

def _key(e : float) -> int:
    "The first four significant figures of the given number"
    return int(10_000*e)

_t_finegrained = {_key(C.e):orbital_time(_s_finegrained)}

def spinor_time(t, C = EotConstants(), e=None):
    """
    Invert t(s), the relationship of orbital time t with the parameter in the spinor
    treatment of the Kepler problem, s, to give s(t).

    Keep a cache of interpolants, one per eccentricity.
    """
    if not e:
        e = C.e
    k = _key(e)
    if k not in _t_finegrained.keys():
        _t_finegrained[k] = orbital_time(_s_finegrained, C, e)
    return np.interp(t, _t_finegrained[k], _s_finegrained)

def orbital_radius(s, C = EotConstants(), e = None):
    "Calculate orbital radial coordinate given spinor time parameter, r(s)"
    A, B, Om, _ = _kepler_params(C, e)
    return A**2 + B**2 + 2*A*B*cos(2*Om*s)

def orbital_angle(s, C = EotConstants(), e = None):
    "Calculate orbital angular coordinate given time parameter, phi(s)"
    A, B, Om, _ = _kepler_params(C, e)
    tanSigY = ( (A**2-B**2)*sin(2*Om*s) )
    tanSigX = ( (A**2+B**2)*cos(2*Om*s) + 2*A*B )
    return np.arctan2( tanSigY, tanSigX ) + pi

