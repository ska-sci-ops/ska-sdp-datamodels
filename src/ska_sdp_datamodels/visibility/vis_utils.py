"""
Geometry support functions.
"""

import numpy
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity

from ska_sdp_datamodels.physical_constants import C_M_S


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


def calculate_transit_time(location, utc_time, direction):
    """Find the UTC time of the nearest transit

    :param location: EarthLocation
    :param utc_time: Time(Iterable)
    :param direction: SkyCoord source
    :return: astropy Time
    """
    site = Observer(location)
    return site.target_meridian_transit_time(
        utc_time, direction, which="next", n_grid_points=100
    )


def get_direction_time_location(vis, time=None):
    """
    Obtain PhaseCentre direction, time,
    and location from Visibility object

    :param vis: Visibility object
    :param time: user-defined time samples;
                 if None, use times from vis
    :return: location, time, direction
            [EarthLocation, UTC Time, PhaseCentre]
    """
    location = vis.configuration.location
    if location is None:
        location = EarthLocation(
            x=Quantity(vis.configuration.antxyz[0]),
            y=Quantity(vis.configuration.antxyz[1]),
            z=Quantity(vis.configuration.antxyz[2]),
        )
    if time is None:
        time = vis.time
    utc_time = Time(time / 86400.0, format="mjd", scale="utc")
    direction = vis.phasecentre
    return location, utc_time, direction


def calculate_visibility_hourangles(vis, time=None):
    """
    Return hour angles for location, utc_time, and direction

    :param vis: Visibility object
    :param time: user-defined time samples;
                 if None, use times from vis
    :return: hour angles
    """
    location, utc_time, direction = get_direction_time_location(vis, time=time)
    site = Observer(location=location)
    hour_angles = site.target_hour_angle(utc_time, direction).wrap_at("180d")
    return hour_angles


def calculate_visibility_uvw_lambda(vis):
    """Recalculate the uvw_lambda values

    :param vis: Visibility
    :return: Visibility with updated uvw_lambda
    """
    k = vis.frequency.data / C_M_S
    uvw_lambda = numpy.einsum("tbs,k->tbks", vis.uvw.data, k)
    vis.visibility_acc.uvw_lambda = uvw_lambda
    return vis
