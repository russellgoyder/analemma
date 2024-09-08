"""
Implementation of formulae for calculating and plotting sundial geometry
"""

import math, numpy as np
from numpy import sin, cos
from scipy import optimize as sci_opt
import datetime
from dataclasses import dataclass, field
from enum import Enum


def hour_angle_terms(alpha, sigma, psi, iota_minus_theta=np.nan):
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

    sinXi_sin_mu = sin(psi) * cos(sigma) * cos(alpha) - cos(psi) * sin(sigma)

    term1 = cos(psi) * cos(sigma) * cos(alpha) + sin(psi) * sin(sigma)
    sinXi_cos_mu = term1 * cos(iota_minus_theta) - cos(sigma) * sin(alpha) * sin(
        iota_minus_theta
    )

    return (sinXi_sin_mu, sinXi_cos_mu)


def hour_angle(alpha, sigma, psi, iota_minus_theta=np.nan):
    "Evaluate the inverse tangent of the sun's hour angle"

    sinXi_sin_mu, sinXi_cos_mu = hour_angle_terms(alpha, sigma, psi, iota_minus_theta)
    return np.arctan2(sinXi_sin_mu, sinXi_cos_mu)


def shadow_denom(alpha, sigma, psi, theta, i, d):
    "The denominator in the shadow coordinate expressions"

    sinXi_sin_mu_s, sinXi_cos_mu_s = hour_angle_terms(
        alpha, sigma, psi
    )  # calc with gnomon as a style
    return (
        sinXi_cos_mu_s * (sin(i) * cos(d) * cos(theta) - sin(theta) * cos(i))
        - sinXi_sin_mu_s * sin(d) * sin(i)
    ) + (sin(i) * sin(theta) * cos(d) + cos(i) * cos(theta)) * sin(alpha) * cos(sigma)


def shadow_coords_xy(alpha, sigma, psi, iota, theta, i, d):
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


pi = math.pi


@dataclass
class PlanetParameters:
    """
    Parameters defining a planet for sundial calculation purposes

    :param N Number of mean days in a year
    :param T_d Number of seconds in a mean day
    :param rho Angle between axes of the ellipse and the equinoxes / solstices
    :param alpha Inclination of the earths axis of rotation
    :param a Length of the planet's orbit's semi-major axis
    :param e Eccentricity of the planet's orbit
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
        self.T_y = self.N * self.T_d  # number of seconds in a mean year
        self.om_y = (
            2 * pi / self.T_y
        )  # mean angular speed of the earth's centre of mass in its orbit
        self.om_d = (
            2 * pi / self.T_d
        )  # angular speed of a point on the earth about the earth's centre of mass
        self.om_sd = (
            (self.N + 1) / self.N * self.om_d
        )  # angular speed of a point on an earth that revolves once per siderial day
        self.T_sd = (
            self.N / (self.N + 1) * self.T_d
        )  # number of seconds in a siderial day
        self.Om = (
            pi / self.T_y * self.a
        )  # angular speed parameter used in spinor orbit formalism ( = om_y / 2 * a )

    @classmethod
    def earth(cls):
        return PlanetParameters(
            N=365.2422,
            T_d=24 * 3600,
            rho=12.25 / 180 * pi,
            alpha=23.5 / 180 * pi,
            a=149598000000,
            e=0.017,
        )


earth = PlanetParameters.earth()


def _kepler_params(planet: PlanetParameters = earth, e=None):
    a = planet.a
    if not e:
        e = planet.e
    b = a * math.sqrt(1 - e**2)  # semi-minor axis
    A = math.sqrt((a + b) / 2)
    B = math.sqrt((a - b) / 2)
    return A, B, planet.Om, planet.T_y


def orbital_time(s, planet: PlanetParameters = earth, e=None):
    "Calculate orbital time given time parameter, t(s)"
    A, B, Om, T_y = _kepler_params(planet, e)
    return (A**2 + B**2) * s + A * B / Om * sin(2 * Om * s) + T_y / 2


_s_finegrained = np.linspace(-pi / earth.Om / 2, pi / earth.Om / 2, 10_000)


def _key(e: float) -> int:
    "The first four significant figures of the given number"
    return int(10_000 * e)


_t_finegrained = {_key(earth.e): orbital_time(_s_finegrained)}


def spinor_time(t, planet: PlanetParameters = earth, e=None):
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


def orbital_radius(s, planet: PlanetParameters = earth, e=None):
    "Calculate orbital radial coordinate given spinor time parameter, r(s)"
    A, B, Om, _ = _kepler_params(planet, e)
    return A**2 + B**2 + 2 * A * B * cos(2 * Om * s)


def orbital_angle(s, planet: PlanetParameters = earth, e=None):
    "Calculate orbital angular coordinate given time parameter, phi(s)"
    A, B, Om, _ = _kepler_params(planet, e)
    tanSigY = (A**2 - B**2) * sin(2 * Om * s)
    tanSigX = (A**2 + B**2) * cos(2 * Om * s) + 2 * A * B
    return np.arctan2(tanSigY, tanSigX) + pi


@dataclass
class DialParameters:
    """
    Parameters defining a sundial

    :param theta 90 degrees minus the latitude of the sundial
    :param iota Inclination of the gnomon
    :param i Inclination of the dial face
    :param d Declination of the dial face
    :x_length Width of the dial face in gnomon lengths
    :y_length Length of the dial face in gnomon lengths
    """

    theta: float
    iota: float
    i: float
    d: float
    x_length: float = 10  # gnomon has unit length
    y_length: float = 10

    def trim_coords(self, x: np.array, y: np.array):
        # set points falling outside the dial face to nan so they don't show up when plotted
        dial_trim = lambda vec, dial_length: np.array(
            [coord if np.abs(coord) < dial_length else np.nan for coord in vec]
        )
        return (dial_trim(x, self.x_length), dial_trim(y, self.y_length))


def _psi(t, planet):
    return np.mod(planet.rho + planet.om_sd * t, 2 * pi)


def _sigma(t, planet):
    phi = orbital_angle(spinor_time(t))
    return np.mod(
        pi + planet.rho + phi, 2 * pi
    )  # phi starts at perihelion, sigma starts at winter solstice


def sin_sunray_dialface_angle(t, planet: PlanetParameters, dial: DialParameters):
    "Sine of the angle between the sun ray and the dial face"

    alpha = planet.alpha
    theta = dial.theta
    i = dial.i
    d = dial.d

    psi = _psi(t, planet)
    sigma = _sigma(t, planet)

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

    Time zero is at perihelion
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

    def sample_times_for_one_day(self, res=1000):
        "Generate an array of times suitable for sampling the progress of the sun across the sky"
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
    days_since_perihelion: int, planet: PlanetParameters, dial: DialParameters
) -> SunTimes:
    """
    Find sunrise, noon and sunset (relative to the dial face)

    Note these because times are all relative to the dial face they do not match the common notions except for an analematic dial
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


def orbit_day_to_date(orbit_day: int, year=2024) -> datetime.date:
    """
    Convert from the number of days since perihelion to the date

    Note that this varies from year to year and this implementation is only exact for 2024
    """
    perihelion_date = datetime.date.fromisoformat(
        f"{year}-01-03"
    )  # approximately true for other years
    return perihelion_date + datetime.timedelta(days=int(orbit_day))


def orbit_date_to_day(the_date: datetime.date, year=2024) -> int:
    """
    Convert from the date to the number of days since perihelion

    Note that this varies from year to year and this implementation is only exact for 2024
    """
    perihelion_date = datetime.date.fromisoformat(
        f"{year}-01-03"
    )  # approximately true for other years
    return (the_date - perihelion_date).days


def sunray_dialface_angle_over_one_year(
    planet: PlanetParameters, dial: DialParameters, hour_offset=0
):
    "Calculate daily the time since perihelion in seconds and the corresponding sin(sunray-dialface angle)"
    times = planet.T_d * np.arange(0, 365)
    times += hour_offset * 3600
    sines = np.array(sin_sunray_dialface_angle(times, planet, dial))
    return (times, sines)


def find_daytime_offsets(planet: PlanetParameters, dial: DialParameters):
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
    "Start in Winter because orbit time starts at perihelion"
    Winter = 0
    Spring = 1
    Summer = 2
    Autumn = 3


def _calc_analemma_points(t: np.array, planet: PlanetParameters, dial: DialParameters):
    psis = _psi(t, planet)
    sigmas = _sigma(t, planet)
    x_raw, y_raw = shadow_coords_xy(
        planet.alpha, sigmas, psis, dial.iota, dial.theta, dial.i, dial.d
    )
    x, y = dial.trim_coords(x_raw, y_raw)
    # when i == d == 0, the x axis (m1) points South and the y axis (m2) points East
    # more natural to rotate clockwise by pi/2 so that x points East and y points North
    xx = y
    yy = -x
    return xx, yy


def _plot_analemma_segment(
    ax,
    times: np.array,
    planet: PlanetParameters,
    dial: DialParameters,
    format_string="-b",
    label="",
    **kwargs,
):

    x, y = _calc_analemma_points(times, planet, dial)
    return ax.plot(x, y, format_string, label=label, **kwargs)


def _analemma_plot_sampling_times(
    season: Season, hour_offset, planet: PlanetParameters, dial: DialParameters
):

    # season lengths are [89, 91, 94, 91] (Winter Spring Summer Autumn)
    # place equinoxes and solstices in the middle for plotting
    season_boundaries = [0, 44, 135, 229, 320]
    if season != Season.Winter:
        start_day = season_boundaries[season.value]
        end_day = season_boundaries[season.value + 1]
        # add 1 to end_day to overlap with first day of next season to avoid gap in plotted line
        times = planet.T_d * np.arange(start_day, end_day + 1)
    else:
        # winter starts on day 320 and wraps to 0 after 364
        times = planet.T_d * (
            np.concatenate(
                (
                    np.arange(season_boundaries[-1], 365),
                    np.arange(0, season_boundaries[1] + 1),
                )
            )
        )

    # note that when hour_offset = 0 we hit mean noon, not solar noon
    # if we adjusted the time to match noon returned by find_sun_rise_noon_set_relative_to_dial_face,
    # then instead of analemmas we would be plotting straight lines
    times += hour_offset * 3600

    ssda = sin_sunray_dialface_angle(times, planet, dial)
    return times[ssda > 0]


_season_format_strings = ["--b", "-g", "-.r", ":k"]


def plot_analemma_season_segment(
    ax,
    season: Season,
    hour_offset: int,
    planet: PlanetParameters,
    dial: DialParameters,
    **kwargs,
):
    "Plot the analemma segment for the given season"

    times = _analemma_plot_sampling_times(season, hour_offset, planet, dial)
    return _plot_analemma_segment(
        ax,
        times,
        planet,
        dial,
        _season_format_strings[season.value],
        label="",
        **kwargs,
    )


@dataclass
class _OrbitDateAndAngle:
    date: datetime.date
    sigma: float


_equinox_or_solstice_info = {
    Season.Summer.value: _OrbitDateAndAngle(
        datetime.date.fromisoformat("2024-06-20"), 0
    ),
    Season.Spring.value: _OrbitDateAndAngle(
        datetime.date.fromisoformat("2024-03-20"), pi / 2
    ),
    Season.Winter.value: _OrbitDateAndAngle(
        datetime.date.fromisoformat("2024-12-21"), pi
    ),
    Season.Autumn.value: _OrbitDateAndAngle(
        datetime.date.fromisoformat("2024-09-22"), 3 * pi / 2
    ),
}


def plot_special_sun_path(
    ax, season: Season, planet: PlanetParameters, dial: DialParameters, **kwargs
):
    "Plot the path of the sun across the dial on the equinox or solstice in the given season"

    orbit_day = orbit_date_to_day(_equinox_or_solstice_info[season.value].date)
    sun_times = find_sun_rise_noon_set_relative_to_dial_face(orbit_day, planet, dial)

    buffer_seconds = 0.1 * 3600
    start_seconds = sun_times.sunrise.absolute_seconds + buffer_seconds
    finish_seconds = sun_times.sunset.absolute_seconds - buffer_seconds

    times = np.linspace(start_seconds, finish_seconds, 1000)
    psis = _psi(times, planet)

    sigma = _equinox_or_solstice_info[season.value].sigma
    x_raw, y_raw = shadow_coords_xy(
        planet.alpha, sigma, psis, dial.iota, dial.theta, dial.i, dial.d
    )

    x, y = dial.trim_coords(x_raw, y_raw)

    # see comment in _calc_analemma_points
    xx = y
    yy = -x

    return ax.plot(
        xx, yy, _season_format_strings[season.value], label=season.name, **kwargs
    )


def _analemma_point_coordinates(
    days_since_perihelion: int,
    hour_offset: int,
    planet: PlanetParameters,
    dial: DialParameters,
):
    solstice_time = np.array([planet.T_d * days_since_perihelion + hour_offset * 3600])
    if sin_sunray_dialface_angle(solstice_time, planet, dial) < 0:
        return np.nan, np.nan
    return _calc_analemma_points(solstice_time, planet, dial)


_sun_times_cache = {}


def _get_sun_times(planet: PlanetParameters, dial: DialParameters):
    key = (id(planet), id(dial))
    if key not in _sun_times_cache.keys():
        _sun_times_cache[key] = [
            find_sun_rise_noon_set_relative_to_dial_face(
                days_since_perihelion, planet, dial
            )
            for days_since_perihelion in np.arange(0, 365)
        ]
    return _sun_times_cache[key]


def _solstice_days(planet: PlanetParameters, dial: DialParameters):
    sun_times = _get_sun_times(planet, dial)
    day_lengths = [
        st.sunset.hours_from_midnight - st.sunrise.hours_from_midnight
        for st in sun_times
    ]
    return (np.argmax(day_lengths), np.argmin(day_lengths))


def _displace_point_along_hour_line(p, away_from_origin):
    L = np.sqrt(p[0] ** 2 + p[1] ** 2)
    direction_factor = 1 if away_from_origin else -1
    factor = 1.5 * direction_factor
    u = (factor * p[0] / L, factor * p[1] / L)
    return (p[0] + u[0], p[1] + u[1])


def _furthest_point(p1, p2):
    d1 = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
    d2 = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
    return p1 if d1 > d2 else p2


def _analemma_label_coordinates(
    hour_offset: int, planet: PlanetParameters, dial: DialParameters
):

    june_solstice_day, december_solstice_day = _solstice_days(planet, dial)

    falls_on_dial = lambda x, y: (
        True if abs(x) <= dial.x_length and abs(y) <= dial.y_length else False
    )

    xj, yj = _analemma_point_coordinates(june_solstice_day, hour_offset, planet, dial)
    june_solstice_falls_on_dial = falls_on_dial(xj, yj)

    xd, yd = _analemma_point_coordinates(
        december_solstice_day, hour_offset, planet, dial
    )
    december_solstice_falls_on_dial = falls_on_dial(xd, yd)

    if june_solstice_falls_on_dial and december_solstice_falls_on_dial:
        p = _furthest_point((xj, yj), (xd, yd))
        return p, _displace_point_along_hour_line(p, away_from_origin=True)

    if not june_solstice_falls_on_dial and not december_solstice_falls_on_dial:
        return None

    if june_solstice_falls_on_dial and not december_solstice_falls_on_dial:
        p = (xj, yj)
        return p, _displace_point_along_hour_line(p, away_from_origin=False)
    elif december_solstice_falls_on_dial and not june_solstice_falls_on_dial:
        p = (xd, yd)
        return p, _displace_point_along_hour_line(p, away_from_origin=True)


def hour_offset_to_oclock(hour_offset: int):
    "Render an integer hour offset (eg +2) as the corresponding time (eg '2pm')"
    if hour_offset == 0:
        return "12pm"
    elif hour_offset == -12:
        return "12am"
    elif hour_offset > 0:
        return f"{hour_offset}pm"
    elif hour_offset < 0:
        return f"{12+hour_offset}am"
    else:
        raise Exception(f"hour_offset {hour_offset} doesn't seem to be a number")


def annotate_analemma_with_hour(
    ax, hour_offset: int, planet: PlanetParameters, dial: DialParameters
):
    "For the given hour, annotate with the time"
    if hour_offset % 3 == 0:
        points = _analemma_label_coordinates(hour_offset, planet, dial)
        if points:
            p, ptext = points
            return ax.annotate(
                hour_offset_to_oclock(hour_offset),
                xy=p,
                xytext=ptext,
                arrowprops={"arrowstyle": "-"},
                horizontalalignment="center",
                fontsize="small",
            )
    return None
