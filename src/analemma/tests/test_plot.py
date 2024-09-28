import datetime
import numpy as np
from analemma import plot as ap


def test_sun_rise_noon_set(earth, camdial):
    st = ap.find_sun_rise_noon_set_relative_to_dial_face(
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
        ap.find_sun_rise_noon_set_relative_to_dial_face(
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

    assert ap.orbit_day_to_date(0) == datetime.date.fromisoformat("2024-01-03")
    assert ap.orbit_day_to_date(june_solstice) == datetime.date.fromisoformat(
        "2024-06-21"
    )
    assert ap.orbit_day_to_date(december_solstice) == datetime.date.fromisoformat(
        "2024-12-21"
    )

    assert ap.orbit_date_to_day(datetime.date.fromisoformat("2024-01-03")) == 0
    assert (
        ap.orbit_date_to_day(datetime.date.fromisoformat("2024-06-21")) == june_solstice
    )
    assert (
        ap.orbit_date_to_day(datetime.date.fromisoformat("2024-12-21"))
        == december_solstice
    )

    arbitrary_date = datetime.date.fromisoformat("2024-05-26")
    assert ap.orbit_day_to_date(ap.orbit_date_to_day(arbitrary_date)) == arbitrary_date
