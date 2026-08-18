"""Moisture content from a hygrometer, for people who own one.

The deterioration model needs moisture content of the grain. A moisture meter
measures it directly; a hygrometer measures the air around it, and equilibrium
moisture relationships convert between the two. Most people storing grain own
the second instrument, so without this the model cannot be reached by the
person it was built for.

The figure this produces is **derived, not measured**, and carries meaningfully
more error than a meter reading. It is returned in a type that says so, rather
than as a bare float that would be indistinguishable downstream from a
measurement.

Source
------
Armstrong, P. R., Casada, M. E., & Lawrence, J. "Development of equilibrium
moisture relationships for storage moisture monitoring of corn." *Applied
Engineering in Agriculture*, USDA-ARS. Modified Chung-Pfost parameters,
desorption, sample USDA1: the widest fitted moisture band of the corn samples
reported, 12-28% dry basis, which contains the 12-14% wet basis regime this
project works in.

That paper cites ASABE D245.6 but does not reproduce its constants, and the
standard itself has not been obtained. See EMC_VERIFIED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Modified Chung-Pfost parameters for shelled corn, desorption.
#: RH = exp[ -(A / (T + C)) * exp(-B * M) ], M in percent dry basis, T in C.
CHUNG_PFOST_A = 523.8
CHUNG_PFOST_B = 0.137
CHUNG_PFOST_C = 119.7

#: False for the same reason CONSTANTS_VERIFIED is false for the dry-matter
#: loss curve: these parameters are one published corn sample's fit, not the
#: ASABE D245.6 standard values. The equation form is not in doubt; the
#: parameters should be replaced once the standard is in hand.
EMC_VERIFIED = False

#: The fitted band. Outside it the equation still returns a number and the
#: number is not evidence.
RH_VALID_MIN = 0.20
RH_VALID_MAX = 0.90

#: How wrong a derived figure can be against a meter, in points of wet basis.
#: Armstrong et al. report standard errors around 1.5 points of dry basis for
#: sensor-based prediction; carried across as a flat figure because the error
#: is dominated by grain-to-grain variation, not by the reading.
DERIVED_UNCERTAINTY_PCT_WB = 1.5


@dataclass(frozen=True)
class DerivedMoisture:
    """Moisture content inferred from air, and the fact that it was inferred.

    Deliberately not a float. A derived moisture that could be passed silently
    wherever a measured one is expected would lose the only thing that
    distinguishes them by the time it reached a displayed number.
    """

    moisture_pct_wb: float
    moisture_pct_db: float
    temperature_c: float
    relative_humidity: float
    uncertainty_pct_wb: float
    derived: bool
    source: str


def moisture_from_humidity(
    temperature_c: float, relative_humidity: float
) -> DerivedMoisture:
    """Equilibrium moisture content of shelled corn for air at these conditions.

    ``relative_humidity`` is a fraction between 0 and 1, not a percentage. The
    two are both plausible things to type and only one is right, so 65 is
    refused rather than read as an out-of-range fraction.
    """
    if not 0.0 < relative_humidity < 1.0:
        raise ValueError(
            f"relative humidity must be a fraction between 0 and 1, "
            f"got {relative_humidity}"
        )
    if not RH_VALID_MIN <= relative_humidity <= RH_VALID_MAX:
        raise ValueError(
            f"relative humidity {relative_humidity:.0%} is outside the fitted "
            f"band {RH_VALID_MIN:.0%}-{RH_VALID_MAX:.0%}; the isotherm is not "
            f"evidence out here"
        )

    inner = -(temperature_c + CHUNG_PFOST_C) * math.log(relative_humidity)
    moisture_db = -(1.0 / CHUNG_PFOST_B) * math.log(inner / CHUNG_PFOST_A)
    moisture_wb = 100.0 * moisture_db / (100.0 + moisture_db)

    return DerivedMoisture(
        moisture_pct_wb=moisture_wb,
        moisture_pct_db=moisture_db,
        temperature_c=temperature_c,
        relative_humidity=relative_humidity,
        uncertainty_pct_wb=DERIVED_UNCERTAINTY_PCT_WB,
        derived=True,
        source=(
            "Armstrong, Casada & Lawrence, Applied Eng. in Agric. (USDA-ARS), "
            "modified Chung-Pfost, corn desorption, sample USDA1"
        ),
    )
