"""
Plot analemmas
"""

import numpy as np
from numpy import sin, cos
from scipy import optimize as sci_opt
from dataclasses import dataclass
import datetime
from enum import Enum
from typing import TypeVar, Tuple
from analemma import geometry, orbit


pi = np.pi


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
        """

        def dial_trim(vec, dial_length):
            return np.array(
                [coord if np.abs(coord) < dial_length else np.nan for coord in vec]
            )

        return dial_trim(x, self.x_length), dial_trim(y, self.y_length)

    def within_face(self, x: np.array, y: np.array):
        """
        TODO
        """

        def _within_dialface(vec, dial_length):
            return np.array(
                [True if np.abs(coord) < dial_length else False for coord in vec]
            )

        return _within_dialface(x, self.x_length) & _within_dialface(y, self.y_length)


def _psi(t, planet):
    return np.mod(planet.rho + planet.om_sd * t, 2 * pi)


def _sigma(t, planet):
    phi = orbit.orbital_angle(orbit.spinor_time(t))
    return np.mod(
        pi + planet.rho + phi, 2 * pi
    )  # phi starts at perihelion, sigma starts at winter solstice


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
    "Start in Winter because orbit time starts at perihelion"

    Winter = 0
    Spring = 1
    Summer = 2
    Autumn = 3


def _calc_raw_analemma_points(
    t: np.array, planet: orbit.PlanetParameters, dial: DialParameters
):
    psis = _psi(t, planet)
    sigmas = _sigma(t, planet)
    x, y = geometry.shadow_coords_xy(
        planet.alpha, sigmas, psis, dial.iota, dial.theta, dial.i, dial.d
    )
    # when i == d == 0, the x axis (m1) points South and the y axis (m2) points East
    # more natural to rotate clockwise by pi/2 so that x points East and y points North
    xx = y
    yy = -x
    return xx, yy, dial.within_face(xx, yy)


def _calc_analemma_points(
    t: np.array, planet: orbit.PlanetParameters, dial: DialParameters, trim: bool = True
):
    x, y, _ = _calc_raw_analemma_points(t, planet, dial)
    return dial.trim_coords(x, y) if trim else (x, y)


Axes = TypeVar("matplotlib.axes.Axes")


def _plot_analemma_segment(
    ax: Axes,
    times: np.array,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    format_string: str = "",
    **kwargs,
):
    x, y = _calc_analemma_points(times, planet, dial)
    return ax.plot(x, y, format_string, **kwargs)


def _analemma_plot_sampling_times(
    season: Season,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
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
    ax: Axes,
    season: Season,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    **kwargs,
):
    """
    Plot the analemma segment for the given season

    Parameters:
        ax: matplotlib axes
        season: The given season
        hour_offset: Number of hours relative to noon, eg -2.25 corresponds to 9:45am
        planet: The planet on which the dial is located
        dial: The orientation and location of the sundial
    """

    times = _analemma_plot_sampling_times(season, hour_offset, planet, dial)
    if times.size == 0:
        return ax
    return _plot_analemma_segment(
        ax,
        times,
        planet,
        dial,
        _season_format_strings[season.value],
        **kwargs,
    )


def plot_analemma(
    ax: Axes,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    format_string: str = "",
    **kwargs,
):
    """
    Plot the analemma

    Parameters:
        ax: matplotlib axes
        hour_offset: Number of hours relative to noon, eg -2.25 corresponds to 9:45am
        planet: The planet on which the dial is located
        dial: The orientation and location of the sundial
    """

    times = planet.T_d * np.arange(0, 365 + 1, dtype=float)
    times += hour_offset * 3600
    ssda = sin_sunray_dialface_angle(times, planet, dial)

    return _plot_analemma_segment(
        ax,
        times[ssda > 0],
        planet,
        dial,
        format_string,
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


class DayType(Enum):
    SunNeverRises = 0
    SunNeverSets = 1
    SunRisesAndSets = 2


def _determine_day_type(
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    orbit_day: int,
):
    hour_offsets = np.arange(-12, 12)
    sines = np.array(
        [sunray_dialface_angle(planet, dial, orbit_day, hour) for hour in hour_offsets]
    )
    if np.all(sines < 0):
        return DayType.SunNeverRises
    elif np.all(sines >= 0):
        return DayType.SunNeverSets
    else:
        return DayType.SunRisesAndSets


def plot_special_sun_path(
    ax: Axes,
    season: Season,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    **kwargs,
):
    """
    Plot the path of the sun across the dial on the equinox or solstice in the given season
    """

    num_times = 1000

    orbit_day = orbit_date_to_day(_equinox_or_solstice_info[season.value].date)
    day_type = _determine_day_type(planet, dial, orbit_day)
    if day_type == DayType.SunNeverRises:
        return []
    elif day_type == DayType.SunNeverSets:
        start_seconds = planet.T_d * orbit_day
        finish_seconds = start_seconds + planet.T_d
        times = np.linspace(start_seconds, finish_seconds, num_times)
    elif day_type == DayType.SunRisesAndSets:
        sun_times = find_sun_rise_noon_set_relative_to_dial_face(
            orbit_day, planet, dial
        )
        buffer_seconds = 0.1 * 3600
        start_seconds = sun_times.sunrise.absolute_seconds + buffer_seconds
        finish_seconds = sun_times.sunset.absolute_seconds - buffer_seconds
        times = np.linspace(start_seconds, finish_seconds, num_times)
    else:
        raise Exception(
            f"Edge case encountered while plotting solstice or equinox for season {season}"
        )

    psis = _psi(times, planet)

    sigma = _equinox_or_solstice_info[season.value].sigma
    x_raw, y_raw = geometry.shadow_coords_xy(
        planet.alpha, sigma, psis, dial.iota, dial.theta, dial.i, dial.d
    )

    x, y = dial.trim_coords(x_raw, y_raw)

    # see comment in _calc_analemma_points
    xx = y
    yy = -x

    return ax.plot(
        xx, yy, _season_format_strings[season.value], label=season.name, **kwargs
    )


def plot_sunrise_sunset(
    ax: Axes,
    date: datetime.date,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
):
    """
    TODO
    """
    orbit_day = orbit_date_to_day(date)
    day_type = _determine_day_type(planet, dial, orbit_day)
    if not day_type == DayType.SunRisesAndSets:
        raise Exception(
            f"Sunrise and sunset events not detected at latitude {pi - dial.theta} on date {date}"
        )

    st = find_sun_rise_noon_set_relative_to_dial_face(orbit_day, planet, dial)

    times = st.sample_times_for_one_day()
    abs_seconds = np.array([st.absolute_seconds for st in times])
    sines = sin_sunray_dialface_angle(abs_seconds, planet, dial)

    ax.plot([st.hours_from_midnight for st in times], sines)
    ax.plot(
        st.sunrise.hours_from_midnight,
        sin_sunray_dialface_angle(st.sunrise.absolute_seconds, planet, dial),
        "sr",
        label="Sunrise",
    )
    ax.plot(
        st.noon.hours_from_midnight,
        sin_sunray_dialface_angle(st.noon.absolute_seconds, planet, dial),
        "og",
        label="Noon",
    )
    ax.plot(
        st.sunset.hours_from_midnight,
        sin_sunray_dialface_angle(st.sunset.absolute_seconds, planet, dial),
        "Db",
        label="Sunset",
    )

    ax.grid()
    ax.set_xlabel("Time in hours since midnight")
    ax.set_ylabel("Sine of sunray-dialface angle")
    ax.set_title(f"Key sundial events on {date}")
    ax.legend()


def plot_annual_sunray_dialface_angle(
    ax1: Axes,
    ax2: Axes,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
):
    """
    TODO
    """

    def _accentuate_x_axis(ax):
        ax.plot([0, 365], [0, 0], "k")

    def _plot_sunray_dialface_angle(
        ax, begin_hour, end_hour, planet: orbit.PlanetParameters, dial: DialParameters
    ):
        for hour_offset in np.arange(begin_hour, end_hour):
            times, sines = sunray_dialface_angle_over_one_year(
                planet, dial, hour_offset
            )
            ax.plot(times / 3600 / 24, sines, label=hour_offset_to_oclock(hour_offset))
        _accentuate_x_axis(ax)
        ax.grid()
        ax.legend()
        ax.set_xlabel("Days since perihelion")

    _plot_sunray_dialface_angle(ax1, -12, 0, planet, dial)
    ax1.set_ylabel("Sine of sunray-dialface angle")
    _plot_sunray_dialface_angle(ax2, 0, 12, planet, dial)


def _analemma_point_coordinates(
    days_since_perihelion: int,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
):
    solstice_time = np.array([planet.T_d * days_since_perihelion + hour_offset * 3600])
    above_dial_plane = sin_sunray_dialface_angle(solstice_time, planet, dial) > 0
    x, y, on_dial = _calc_raw_analemma_points(solstice_time, planet, dial)
    return x, y, on_dial, above_dial_plane


_sun_times_cache = {}


def _get_sun_times(planet: orbit.PlanetParameters, dial: DialParameters):
    key = (id(planet), id(dial))
    if key not in _sun_times_cache.keys():
        _sun_times_cache[key] = [
            find_sun_rise_noon_set_relative_to_dial_face(
                days_since_perihelion, planet, dial
            )
            for days_since_perihelion in np.arange(0, 365)
        ]
    return _sun_times_cache[key]


def longest_and_shortest_days(planet: orbit.PlanetParameters, dial: DialParameters):
    """
    TODO
    """
    sun_times = _get_sun_times(planet, dial)
    day_lengths = [
        st.sunset.hours_from_midnight - st.sunrise.hours_from_midnight
        for st in sun_times
    ]
    return (np.argmax(day_lengths), np.argmin(day_lengths))


def _solstice_days(planet: orbit.PlanetParameters, dial: DialParameters):
    _, sines = sunray_dialface_angle_over_one_year(planet, dial)
    return (np.argmax(sines), np.argmin(sines))


def _point_and_text_coords(j, d, label_point, above_dial_plane):
    j_above_dial_plane, d_above_dial_plane = above_dial_plane
    if label_point == WhichSolstice.June:
        p = j
        q = d
        f = 1.0 if d_above_dial_plane else -1.0
    elif label_point == WhichSolstice.December:
        p = d
        q = j
        f = 1.0 if j_above_dial_plane else -1.0
    else:
        raise Exception(
            "Logical inconsistency: Solstice label point must either \
                        be June or December"
        )

    u = ((p[0] - q[0]), (p[1] - q[1]))
    L = np.sqrt(u[0] ** 2 + u[1] ** 2)
    u /= L
    u *= 0.75 * f

    return p, (p[0] + u[0], p[1] + u[1])


class WhichSolstice(Enum):
    June = 0
    December = 1


def _first_point_is_furthest(p1, p2):
    d1 = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
    d2 = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
    return d1 > d2


def _label_point(
    june_point,
    dec_point,
    june_solstice_falls_on_dial: bool,
    december_solstice_falls_on_dial: bool,
):
    if june_solstice_falls_on_dial and december_solstice_falls_on_dial:
        val = (
            WhichSolstice.June
            if _first_point_is_furthest(june_point, dec_point)
            else WhichSolstice.December
        )
        return val
    elif june_solstice_falls_on_dial and not december_solstice_falls_on_dial:
        return WhichSolstice.June
    elif december_solstice_falls_on_dial and not june_solstice_falls_on_dial:
        return WhichSolstice.December
    else:
        raise Exception(
            f"Logical inconsistency encountered while determining label point with \
                June coords {june_point}, December coords {dec_point}, and J/D falling on \
                    dial: {june_solstice_falls_on_dial}/{december_solstice_falls_on_dial}"
        )


def _analemma_label_coordinates(
    hour_offset: float, planet: orbit.PlanetParameters, dial: DialParameters
):
    june_solstice_day, december_solstice_day = _solstice_days(planet, dial)

    xj, yj, j_on_dial, j_above_dial_plane = _analemma_point_coordinates(
        june_solstice_day, hour_offset, planet, dial
    )
    june_solstice_falls_on_dial = j_on_dial and j_above_dial_plane

    xd, yd, d_on_dial, d_above_dial_plane = _analemma_point_coordinates(
        december_solstice_day, hour_offset, planet, dial
    )
    december_solstice_falls_on_dial = d_on_dial and d_above_dial_plane

    if not june_solstice_falls_on_dial and not december_solstice_falls_on_dial:
        return None

    label_point = _label_point(
        (xj, yj), (xd, yd), june_solstice_falls_on_dial, december_solstice_falls_on_dial
    )

    return _point_and_text_coords(
        (xj, yj), (xd, yd), label_point, (j_above_dial_plane, d_above_dial_plane)
    )


def hour_offset_to_oclock(hour_offset: int):
    """
    Render an integer hour offset (eg +2) as the corresponding time (eg '2pm')

    Note that any non-integer part of the hour offset will be truncated
    """
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
    ax: Axes, hour_offset: int, planet: orbit.PlanetParameters, dial: DialParameters
):
    """
    For the given hour, annotate with the time

    Note that any non-integer part of the hour offset will be truncated
    """
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


def plot_hourly_analemmas(
    ax: Axes,
    planet: orbit.PlanetParameters,
    dial: DialParameters,
    title: str = None,
    **kwargs,
):
    """
    TODO
    """
    hour_offsets = find_daytime_offsets(planet, dial)

    lines_for_legend = []
    seasons = []
    count = 0
    for season in Season:
        for hour in hour_offsets:
            plot_analemma_season_segment(
                ax, season, hour, planet, dial, linewidth=0.75, **kwargs
            )
            annotate_analemma_with_hour(ax, hour, planet, dial)
        lines = plot_special_sun_path(
            ax, season, planet, dial, linewidth=0.75, **kwargs
        )
        if len(lines) > 0:
            lines_for_legend += lines
            seasons += [count]
            count += 1

    # put a circle at the base of the gnomon
    ax.plot(0, 0, "ok")

    # reorder seasons
    ordered_lines = [lines_for_legend[s] for s in seasons]
    ax.legend(handles=ordered_lines)

    if title:
        ax.set_title(title)
