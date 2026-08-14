"""Published constants for the maize deterioration model.

Every value here is traceable to a primary source. The docstring beside each
constant names that source, because a number without a citation cannot be
defended in a paper.

Primary sources
---------------
Steele, J. L. (1967). *Deterioration of damaged shelled corn as measured by
carbon dioxide production.* PhD dissertation, Iowa State University.
Handle 20.500.12876/75955. Open access; local copy in ../../papers/.
Appendix D carries the multiplier equations; the methods section carries the
definition of mechanical damage.

Thompson, T. L. (1972). "Temporary storage of high-moisture shelled corn using
continuous aeration." *Transactions of the ASAE* 15(2): 333-337.
DOI 10.13031/2013.37900. Paywalled; not yet obtained.

Bern, C. J., Steele, J. L., & Morey, R. V. (2002). "Shelled corn CO2 evolution
and storage time for 0.5% dry matter loss." *Applied Engineering in
Agriculture* 18(6).
"""

# --- Dry matter loss curve -------------------------------------------------
# DML = DML_A * (exp(DML_B * tr) - 1) + DML_C * tr,  DML in percent, tr in hours.
#
# UNVERIFIED. These three coefficients come from the Kaleta & Gornicki review's
# rendering of Thompson (1972), not from Thompson's own paper, which has not
# been obtained. They are the only constants in this module still in that
# state. See CONSTANTS_VERIFIED below.
DML_A = 0.0883
DML_B = 0.006
DML_C = 0.00102

#: Spoilage threshold in percent dry matter loss. Bern et al. (2002) frame
#: shelled-corn storage time explicitly around this figure.
DML_THRESHOLD_PCT = 0.5

#: Complete oxidation of carbohydrate: a 1.0% dry matter loss corresponds to
#: this much evolved CO2. Steele (1967), citing Saul & Lind (1958). Used only
#: if CO2 instrumentation is ever added; nothing in the ranking path needs it.
CO2_G_PER_KG_DM_AT_1PCT_DML = 14.7


# --- Reference conditions --------------------------------------------------
# All three multipliers equal 1.0 at these values. Verified numerically on
# 2026-08-15 by evaluating the published forms: MM = 0.9966, MT = 1.0020,
# MD = 1.0230.
#
# The trap this closes: the multipliers are normalised to 30% damage, NOT to
# zero damage. An implementation that sets MD = 1 at zero damage is wrong
# across the whole input range and produces plausible-looking output while
# doing so. At zero damage the correct multiplier is MD_A == 2.08.
REFERENCE_TEMPERATURE_C = 15.6      # 60 degF in Steele's original units
REFERENCE_MOISTURE_PCT_WB = 25.0
REFERENCE_DAMAGE_PCT = 30.0


# --- Temperature multiplier (Steele 1967, Appendix D) ----------------------
# Stated by Steele in degF as 32.3 * exp(-3.48 * (T_F / 60)). Converting,
# (1.8 * T_C + 32) / 60 == 0.03 * T_C + 0.5333, which reproduces the Celsius
# form found in the secondary literature exactly. VERIFIED 2026-08-15.
MT_A = 32.3
MT_B = -3.48
MT_C_SLOPE = 0.03
MT_C_INTERCEPT = 0.5333333333333333   # 32/60, kept exact rather than rounded

#: Correction terms for the warm, damp branches of the piecewise form.
MT_WET_SLOPE = 0.01
MT_WET_CAP = 0.09
MT_WET_EXP_A = 0.61


# --- Moisture multiplier (Steele 1967, Appendix D) -------------------------
# MM = 0.103 * (exp(455 * M_db_pct ** -1.53) - 0.00845 * M_db_pct + 1.558)
# where M_db_pct is moisture on a DRY basis expressed as a percentage.
# VERIFIED 2026-08-15.
MM_A = 0.103
MM_B = 455.0
MM_C = -1.53
MM_D = 0.00845
MM_E = 1.558


# --- Mechanical damage multiplier (Steele 1967, Appendix D) ----------------
# Steele gives one exponential per dry-matter-loss level. The 0.5% form is the
# one that matches this project's threshold.
#
# Do NOT substitute the quadratic 0.001*D**2 - 0.1101*D + 3.426 that circulates
# in the review literature: Kaleta & Gornicki fitted that themselves from
# Steele's data, and it runs up to 65% high at low damage -- exactly the regime
# dried bagged maize occupies. Both are monotone decreasing, so lot ranking is
# unaffected, but absolute day counts are not.
MD_A = 2.08
MD_B = -0.0239

#: The other two levels, kept for completeness and for sensitivity analysis.
MD_COEFFICIENTS_BY_DML_LEVEL = {
    0.1: (1.82, -0.0143),
    0.5: (MD_A, MD_B),
    1.0: (2.17, -0.0254),
}


# --- Validity ranges -------------------------------------------------------
#: Steele states the moisture multiplier for 13% <= M <= 35%, WET basis.
#:
#: The basis is worth pinning down, because the review literature states the
#: same range as 0.149 <= M <= 0.538 kg water / kg dry matter and it is easy to
#: assume the two disagree. They do not: 13% w.b. is 0.1494 kg/kg d.b. and 35%
#: w.b. is 0.5385 kg/kg d.b. Identical range, two bases. Confirmed 2026-08-15.
#:
#: Note that the multiplier's own argument is dry-basis percent -- only this
#: validity range is quoted wet basis.
MM_VALID_WB_PCT = (13.0, 35.0)

#: Damage multiplier range. Below 2% the fit is unsupported by data; above 40%
#: likewise. Note the fitted parabola of the review's re-fit turns at 55%,
#: which is one more reason not to use it.
MD_VALID_PCT = (2.0, 40.0)

#: Steele's tests spanned 35-120 degF.
MT_VALID_C = (1.7, 48.9)


# --- The gate --------------------------------------------------------------
#: Absolute days-to-threshold is suppressed while this is False. Ranking is
#: unaffected and ships regardless, because a ranking needs only that the
#: damage multiplier be monotone in damage -- which is true for every candidate
#: form of the equation.
#:
#: Flip to True only when DML_A, DML_B and DML_C have been checked against
#: Thompson (1972) itself. Nothing else in the codebase may set this.
CONSTANTS_VERIFIED = False

#: Why it is still False, surfaced in the research view so a reviewer or
#: examiner can see the reasoning rather than guess at it.
CONSTANTS_VERIFIED_NOTE = (
    "Thompson (1972) has not been obtained. The three dry-matter-loss "
    "coefficients are transcribed from a review. The moisture, temperature "
    "and mechanical-damage multipliers are verified against Steele (1967), "
    "Appendix D."
)
