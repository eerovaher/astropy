# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for custom error and warning messages in `astropy.coordinates`."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest

from astropy import units as u
from astropy.coordinates import (
    GCRS,
    NonRotationTransformationError,
    NonRotationTransformationWarning,
    SkyCoord,
)

if TYPE_CHECKING:
    from astropy.coordinates import BaseCoordinateFrame


class FrameDescription(NamedTuple):
    frame: BaseCoordinateFrame | SkyCoord
    description: str
    pytest_id: str


gcrs_custom = FrameDescription(
    GCRS(
        0 * u.deg,
        0 * u.deg,
        obstime="J1950",
        obsgeovel=[30, -7, 11] * u.km / u.s,
    ),
    (
        "GCRS Frame (obstime=J1950.000, obsgeoloc=(0., 0., 0.) m, "
        "obsgeovel=(30000., -7000., 11000.) m / s)"
    ),
    "custom_GCRS",
)
gcrs_default = FrameDescription(
    GCRS(0 * u.deg, 0 * u.deg),
    (
        "GCRS Frame (obstime=J2000.000, obsgeoloc=(0., 0., 0.) m, "
        "obsgeovel=(0., 0., 0.) m / s)"
    ),
    "default_GCRS",
)
skycoord_galactic = FrameDescription(
    SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic"),
    "Galactic Frame",
    "SkyCoord_Galactic",
)
skycoord_icrs = FrameDescription(
    SkyCoord(0 * u.deg, 0 * u.deg), "ICRS Frame", "SkyCoord_ICRS"
)


@pytest.mark.parametrize(
    "coord_from,coord_to",
    [(skycoord_icrs, gcrs_custom), (gcrs_default, skycoord_galactic)],
    ids=lambda x: x.pytest_id,
)
def test_NonRotationTransformationError_message(coord_from, coord_to):
    assert str(NonRotationTransformationError(coord_to.frame, coord_from.frame)) == (
        f"refusing to transform other coordinates from <{coord_from.description}> to "
        f"<{coord_to.description}> because angular separation can depend on the "
        "direction of the transformation"
    )


@pytest.mark.parametrize(
    "coord_from,coord_to",
    [(skycoord_icrs, gcrs_default), (gcrs_custom, skycoord_galactic)],
    ids=lambda x: x.pytest_id,
)
def test_NonRotationTransformationWarning_message(coord_from, coord_to):
    assert str(NonRotationTransformationWarning(coord_to.frame, coord_from.frame)) == (
        f"transforming other coordinates from <{coord_from.description}> to "
        f"<{coord_to.description}>. Angular separation can depend on the direction of "
        "the transformation."
    )
