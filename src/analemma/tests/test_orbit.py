import pytest
import datetime
from skyfield import almanac
from skyfield.api import load
from skyfield import searchlib as sf_search


def _ephemeris():
    return load("de440s.bsp")


@pytest.fixture
def ephemeris():
    return _ephemeris()


@pytest.fixture
def timescale():
    return load.timescale()


def test_skyfield_solstices_and_equinoxes(ephemeris, timescale):
    """
    Check the Skyfield package against known season events
    """
    jan1 = timescale.utc(2024, 1, 1)
    dec31 = timescale.utc(2024, 12, 31)
    event_times, _ = almanac.find_discrete(jan1, dec31, almanac.seasons(ephemeris))

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


_eph = _ephemeris()


def _earth_distance(time):
    earth = _eph["earth"]
    sun = _eph["sun"]
    e = earth.at(time)
    s = e.observe(sun)
    return s.distance().km


_earth_distance.step_days = 1


def test_perihelion_date(timescale):
    year = 2024
    start_time = timescale.utc(year, 1, 1)
    end_time = timescale.utc(year + 1, 1, 1)
    perihelion = sf_search.find_minima(start_time, end_time, _earth_distance)
    assert perihelion[0][0].utc_datetime().date() == datetime.date.fromisoformat(
        "2024-01-03"
    )
