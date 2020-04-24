# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A coordinate or frame in the TEME (True Equator, Mean Equinox) system.

TEME is a True equator, Mean Equinox coordinate frames used in NORAD TLE
satellite files.
These are distinct from the ICRS and AltAz frames because they are just
rotations without aberration corrections or offsets.
"""
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.utils.decorators import format_doc
from astropy.coordinates.representation import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc
from astropy.coordinates.attributes import TimeAttribute
from .utils import DEFAULT_OBSTIME

__all__ = ['TEME']

doc_footer_teme = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the frame is defined.  Used for determining the
        position of the Earth.
"""


@format_doc(base_doc, components="", footer=doc_footer_teme)
class TEME(BaseCoordinateFrame):
    """
    A coordinate or frame in the True Equator Mean Equinox frame (TEME).

    This frame is a geocentric system similar to CIRS or geocentric apparent place,
    except that the mean sidereal time is used to rotate from TIRS. TEME coordinates
    are most often used in combination with orbital data for satellites in the
    two-line-ephemeris format.

    For more background on TEME, see the references provided in the
    :ref:`astropy-coordinates-seealso` section of the documentation.
    """

    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential

    obstime = TimeAttribute(default=DEFAULT_OBSTIME)

# Transformation functions for getting to/from TEME and ITRS are in
# intermediate rotation transforms.py
