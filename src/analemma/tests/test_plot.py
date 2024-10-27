import datetime
import numpy as np
from skyfield import almanac
from skyfield.api import load
from analemma import geometry as geom


def test_sun_rise_noon_set(earth, camdial):
    st = geom.find_sun_rise_noon_set_relative_to_dial_face(
        days_since_perihelion=50, planet=earth, dial=camdial
    )
    assert (
        st.sunrise.hours_from_midnight
        < st.noon.hours_from_midnight
        < st.sunset.hours_from_midnight
    )


def test_solstices(earth, camdial):
    """
    The June and December solstices should occur on the longest and shortest day of the year as seen on a horizontal dial
    """

    sun_times = [
        geom.find_sun_rise_noon_set_relative_to_dial_face(
            days_since_perihelion, earth, camdial
        )
        for days_since_perihelion in np.arange(0, 365)
    ]

    day_lengths = [
        st.sunset.hours_from_midnight - st.sunrise.hours_from_midnight
        for st in sun_times
    ]
    december_solstice = np.argmin(day_lengths)
    june_solstice = np.argmax(day_lengths)

    assert geom.orbit_day_to_date(0) == datetime.date.fromisoformat("2024-01-03")
    assert geom.orbit_day_to_date(june_solstice) == datetime.date.fromisoformat(
        "2024-06-21"
    )
    assert geom.orbit_day_to_date(december_solstice) == datetime.date.fromisoformat(
        "2024-12-21"
    )

    assert geom.orbit_date_to_day(datetime.date.fromisoformat("2024-01-03")) == 0
    assert (
        geom.orbit_date_to_day(datetime.date.fromisoformat("2024-06-21"))
        == june_solstice
    )
    assert (
        geom.orbit_date_to_day(datetime.date.fromisoformat("2024-12-21"))
        == december_solstice
    )

    arbitrary_date = datetime.date.fromisoformat("2024-05-26")
    assert (
        geom.orbit_day_to_date(geom.orbit_date_to_day(arbitrary_date)) == arbitrary_date
    )


def test_skyfield_solstices_and_equinoxes():
    """
    Check the Skyfield package against known season events
    """
    ts = load.timescale()
    eph = load("de421.bsp")

    boy = ts.utc(2024, 1, 1)
    eoy = ts.utc(2024, 12, 31)
    event_times, _ = almanac.find_discrete(boy, eoy, almanac.seasons(eph))

    def _assert_season_event(event_index, event_name, event_date_iso_str):
        assert almanac.SEASON_EVENTS_NEUTRAL[event_index] == event_name
        assert (
            datetime.date.fromisoformat(event_date_iso_str)
            == event_times[event_index].utc_datetime().date()
        )

    for n, name_and_date in enumerate(
        zip(
            [
                "March Equinox",
                "June Solstice",
                "September Equinox",
                "December Solstice",
            ],
            ["2024-03-20", "2024-06-20", "2024-09-22", "2024-12-21"],
        )
    ):
        name, date = name_and_date
        _assert_season_event(n, name, date)
