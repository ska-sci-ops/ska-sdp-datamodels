"""
Geometry support functions.
"""

from astroplan import Observer


def generate_baselines(nant):
    """Generate mapping from antennas to baselines
    Note that we need to include auto-correlations
    since some input measurement sets
    may contain auto-correlations

    :param nant: Number of antennas
    """
    for ant1 in range(0, nant):
        for ant2 in range(ant1, nant):
            yield ant1, ant2


def calculate_transit_time(location, utc_time, direction, fraction_day=1e-7):
    """Find the UTC time of the nearest transit

    :param fraction_day: Step in this fraction of day to find transit
    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Time
    """
    site = Observer(location)
    return site.target_meridian_transit_time(
        utc_time, direction, which="next", n_grid_points=100
    )
