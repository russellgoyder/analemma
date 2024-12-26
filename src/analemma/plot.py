"""
Functionality for visualizing analemma projections on sundial and related phenomena
"""

import numpy as np
import datetime
from enum import Enum
from typing import TypeVar
from analemma import geometry as geom, orbit


pi = np.pi


Axes = TypeVar("matplotlib.axes.Axes")


def _plot_analemma_segment(
    ax: Axes,
    times: np.array,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
    format_string: str = "",
    **kwargs,
):
    x, y = geom.calc_analemma_points(times, planet, dial)
    return ax.plot(x, y, format_string, **kwargs)


def _analemma_plot_sampling_times(
    season: geom.Season,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
):
    # season lengths are [89, 92, 94, 90] (Winter Spring Summer Autumn) in 2024
    # place equinoxes and solstices in the middle for plotting
    season_boundaries = [0, 44, 135, 229, 320]
    if season != geom.Season.Winter:
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

    ssda = geom.sin_sunray_dialface_angle(times, planet, dial)
    return times[ssda > 0]


_season_format_strings = ["--b", "-g", "-.r", ":k"]


def plot_analemma_season_segment(
    ax: Axes,
    season: geom.Season,
    hour_offset: float,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
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
        return []
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
    dial: geom.DialParameters,
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
    ssda = geom.sin_sunray_dialface_angle(times, planet, dial)

    return _plot_analemma_segment(
        ax,
        times[ssda > 0],
        planet,
        dial,
        format_string,
        **kwargs,
    )


class DayType(Enum):
    SunNeverRises = 0
    SunNeverSets = 1
    SunRisesAndSets = 2


def _determine_day_type(
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
    orbit_day: int,
):
    hour_offsets = np.arange(-12, 12)
    sines = np.array(
        [
            geom.sunray_dialface_angle(planet, dial, orbit_day, hour)
            for hour in hour_offsets
        ]
    )
    if np.all(sines < 0):
        return DayType.SunNeverRises
    elif np.all(sines >= 0):
        return DayType.SunNeverSets
    else:
        return DayType.SunRisesAndSets


def plot_season_event_sun_path(
    ax: Axes,
    season: geom.Season,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
    year: int = None,
    **kwargs,
):
    """
    Plot the path of the sun across the dial on the equinox or solstice in the given season

    Parameters:
        ax: matplotlib axes
        season: The given season
        planet: The planet on which the dial is located
        dial: The orientation and location of the sundial
        year: The year in which the seasons events fall (defaults to current year)
    """

    num_times = 1000

    if not year:
        year = datetime.date.today().year

    season_event = orbit.season_event_info(season.value, year)

    # for an equatorial dial on the equinoxes, the sun ray is parallel to the dial face
    if (
        season.name in ("Spring", "Autumn")
        and abs(dial.d) < 1.0e-5
        and abs(dial.theta - dial.i) < 0.25 / 180 * pi
    ):
        return []

    orbit_day = orbit.orbit_date_to_day(season_event.date)
    day_type = _determine_day_type(planet, dial, orbit_day)
    if day_type == DayType.SunNeverRises:
        return []
    elif day_type == DayType.SunNeverSets:
        start_seconds = planet.T_d * orbit_day
        finish_seconds = start_seconds + planet.T_d
        times = np.linspace(start_seconds, finish_seconds, num_times)
    elif day_type == DayType.SunRisesAndSets:
        sun_times = geom.find_sun_rise_noon_set_relative_to_dial_face(
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

    psis = planet.rotation_angle(times)

    sigma = season_event.sigma
    x_raw, y_raw = geom.shadow_coords_xy(
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
    dial: geom.DialParameters,
):
    r"""
    Visualize sunrise and sunset relative to a sundial

    This function adds a line and three points to the given axes. The line is the sine of the angle between
    sun rays the face of the sundial, over the course of a day. The three points mark sunrise, noon, and sunset,
    when that angle is zero, maximal, and $\pi$ respectively.

    Parameters:
        ax: matplotlib axes
        date: The date for which to visual sunrise and sunset
        planet: The planet on which the dial is located
        dial: The orientation and location of the sundial
    """
    orbit_day = orbit.orbit_date_to_day(date)
    day_type = _determine_day_type(planet, dial, orbit_day)
    if not day_type == DayType.SunRisesAndSets:
        raise Exception(
            f"Sunrise and sunset events not detected at latitude {pi - dial.theta} on date {date}"
        )

    st = geom.find_sun_rise_noon_set_relative_to_dial_face(orbit_day, planet, dial)

    times = st.sample_times_for_one_day()
    abs_seconds = np.array([st.absolute_seconds for st in times])
    sines = geom.sin_sunray_dialface_angle(abs_seconds, planet, dial)

    ax.plot([st.hours_from_midnight for st in times], sines)
    ax.plot(
        st.sunrise.hours_from_midnight,
        geom.sin_sunray_dialface_angle(st.sunrise.absolute_seconds, planet, dial),
        "sr",
        label="Sunrise",
    )
    ax.plot(
        st.noon.hours_from_midnight,
        geom.sin_sunray_dialface_angle(st.noon.absolute_seconds, planet, dial),
        "og",
        label="Noon",
    )
    ax.plot(
        st.sunset.hours_from_midnight,
        geom.sin_sunray_dialface_angle(st.sunset.absolute_seconds, planet, dial),
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
    dial: geom.DialParameters,
):
    """
    Plot the sine of the sunray-dialface angle for each hour in the day over one year

    If the sine of the angle between the sun ray and the dial face is greater than zero, the gnomon's
    shadow may fall on the dial (depending on its size) and therefore part of the analemma may be
    visible. This defines daytime relative to the dial.

    Parameters:
        ax1: A matplotlib axes object to hold plots for the morning hours
        ax2: A matplotlib axes object to hold plots for the afternon and evening hours
        planet: The planet on which the dial is located
        dial: The orientation and location of the sundial
    """

    def _accentuate_x_axis(ax):
        ax.plot([0, 365], [0, 0], "k")

    def _plot_sunray_dialface_angle(
        ax,
        begin_hour,
        end_hour,
        planet: orbit.PlanetParameters,
        dial: geom.DialParameters,
    ):
        for hour_offset in np.arange(begin_hour, end_hour):
            times, sines = geom.sunray_dialface_angle_over_one_year(
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
    dial: geom.DialParameters,
):
    solstice_time = np.array([planet.T_d * days_since_perihelion + hour_offset * 3600])
    above_dial_plane = geom.sin_sunray_dialface_angle(solstice_time, planet, dial) > 0
    x, y, on_dial = geom.calc_raw_analemma_points(solstice_time, planet, dial)
    return x, y, on_dial, above_dial_plane


def _solstice_days(planet: orbit.PlanetParameters, dial: geom.DialParameters):
    _, sines = geom.sunray_dialface_angle_over_one_year(planet, dial)
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
    hour_offset: float, planet: orbit.PlanetParameters, dial: geom.DialParameters
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


def _font_size(ax: Axes) -> str:
    "Map bounding box width in inches to font size string"
    bbox = ax.get_window_extent().transformed(
        ax.get_figure().dpi_scale_trans.inverted()
    )
    if bbox.width >= 6:
        return "small"
    elif bbox.width >= 4:
        return "x-small"
    else:
        return "xx-small"


def annotate_analemma_with_hour(
    ax: Axes,
    hour_offset: int,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
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
                fontsize=_font_size(ax),
            )
    return None


def _reorder_legend_info(legend_info):
    """
    Move Winter to the end of the list
    """
    if legend_info[0][1] == "Winter":
        winter_entry = legend_info.pop(0)
        legend_info.append(winter_entry)
    return legend_info


def _adjust_for_hemisphere(dial, labels):
    if dial.theta < pi / 2:
        return labels
    else:
        mapping = {
            "Winter": "Summer",
            "Spring": "Autumn",
            "Summer": "Winter",
            "Autumn": "Spring",
        }
        return [mapping[label] for label in labels]


def plot_hourly_analemmas(
    ax: Axes,
    planet: orbit.PlanetParameters,
    dial: geom.DialParameters,
    title: str = None,
    year: int = None,
    **kwargs,
):
    """
    Plot one analemma for each hour as seen on the face of a sundial

    This function plots several analemmas, one per hour of daytime. The line style shows the season. One line showing
    the path of the shadow tip during the day for each solstice is also shown (with line style appropriate to the
    season) and on a horizontal dial forms an envelope marking the longest shadows in Winter and the shortest shadows in
    Summer. Similarly, the path of the shadow tip on each equinox is shown and appears as a straight line. Moreover, the
    two straight lines fall on top of each other.

    In the legend, the seasons are labelled according to the hemisphere in which the sundial is located, so that for
    sundials in the southern hemisphere, summer occurs in December.

    Parameters:
        ax: matplotlib axes planet: The planet on which the dial is located dial: The orientation and location of the
        sundial title: Title to add to the axes year: Year for which the plot hourly analemmas (defaults to current
        year)
    """
    hour_offsets = geom.find_daytime_offsets(planet, dial)

    legend_info = []
    for season in geom.Season:
        segment_lines = []
        for hour in hour_offsets:
            segment_lines += plot_analemma_season_segment(
                ax, season, hour, planet, dial, linewidth=0.75, **kwargs
            )
            annotate_analemma_with_hour(ax, hour, planet, dial)

        if len(segment_lines) > 0:
            legend_info.append((segment_lines[0], season.name))

        plot_season_event_sun_path(
            ax, season, planet, dial, linewidth=0.75, year=year, **kwargs
        )

    # put a circle at the base of the gnomon
    ax.plot(0, 0, "ok")

    handles, labels = zip(*_reorder_legend_info(legend_info))
    labels = _adjust_for_hemisphere(dial, labels)
    ax.legend(handles, labels, fontsize=_font_size(ax))

    if title:
        ax.set_title(title)
