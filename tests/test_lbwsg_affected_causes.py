"""Blind verification checks for MIC-7320 -- separate LBWSG-affected causes.

These InteractiveContext checks assert the *quantitative expectations* named in the
MIC-7320 iteration plan (the contract), NOT the implementation. They are authored to be
run *later*, once the artifact has been rebuilt with the new affected-causes mortality-risk
key. Until then every check skips cleanly (see ``require_affected_causes_artifact``).

Plan expectations verified here (see mic-7320-iteration-plan.md "Quantitative expectations"):

  1. AFFECTED-cause neonatal mortality (preterm / sepsis / encephalopathy) VARIES across
     LBWSG exposure categories.
     -> test_affected_cause_mortality_varies_by_lbwsg_category
  2. UNAFFECTED ("other_causes") neonatal mortality VARIES across LBWSG exposure categories.
     -> test_unaffected_cause_mortality_varies_by_lbwsg_category
  3. UNAFFECTED ("other_causes") neonatal mortality does NOT change when an LBWSG-affecting
     intervention (IFA/MMS or IV iron) coverage is added.
     -> test_unaffected_mortality_invariant_to_intervention_coverage
  4. AFFECTED-cause mortality changes under added intervention coverage.
     -> test_affected_mortality_changes_with_intervention_coverage
  5. RR of neonatal death at specific LBWSG categories is within 10% of the GBD-derived
     ratio (the artifact LBWSG relative risk).
     -> test_relative_risk_of_neonatal_death_matches_gbd_within_10pct

Notes / caveats surfaced to the caller (see the escalations in the agent report):
  * The plan equates observable ``child_cause_of_death == "other_causes"`` with the
    algebraic ``ACMRisk_unaffected`` term. They are NOT identical: per the plan's own
    decomposition, sim ``other_causes`` still contains the *non-modeled AFFECTED* causes
    (diarrhea, LRI, meningitis, hemolytic disease, other neonatal, SIDS), which use the
    scenario RR and therefore DO respond a little to intervention coverage. Expectation (3)
    is therefore encoded as "changes far less than the affected causes" rather than "does
    not change at all". A cleaner check would assert against a dedicated
    ``ACMRisk_unaffected`` pipeline, but the contract does not name one.
  * Only ``mms_total_scaleup`` is exercised as the LBWSG-affecting coverage lever. The plan
    lists "IFA/MMS or IV iron"; ``anemia_screening_and_iv_iron_scaleup`` is an equally valid
    lever and can be added by parametrizing ``INTERVENTION_SCENARIO_NAME``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium.artifact import Artifact
from vivarium.engine import InteractiveContext
from vivarium.testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import LBWSG
from vivarium_gates_mncnh.constants.data_values import (
    CAUSES_OF_NEONATAL_MORTALITY,
    COLUMNS,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS

from .test_lbwsg import get_simulation_exposure_categories
from .utilities import get_births_and_deaths_idx, get_interactive_context_state

# ``build_model_specification`` lets us override the intervention scenario before setup so we
# can build a baseline sim and an added-coverage sim from the same spec (common random
# numbers). Imported defensively so that a differing engine layout skips rather than errors.
try:
    from vivarium.engine.framework.configuration import build_model_specification

    _HAS_BUILD_SPEC = True
except ImportError:  # pragma: no cover - environment guard
    build_model_specification = None
    _HAS_BUILD_SPEC = False


# Plan working name for the new artifact key (constants/data_keys.py:
# POPULATION.AFFECTED_CAUSES_MORTALITY_RISK). Referenced as a literal so this module imports
# even before the constant/key exist; the require_* fixture skips until the rebuilt artifact
# actually contains it. If the finalized key name differs, update this in one place.
AFFECTED_CAUSES_MORTALITY_RISK_KEY = "cause.all_causes.affected_causes_mortality_risk"

# The LBWSG-affecting coverage lever used for the baseline-vs-intervention comparison.
INTERVENTION_SCENARIO_NAME = INTERVENTION_SCENARIOS.MMS_TOTAL_SCALEUP.name

# Causes the plan classifies as AFFECTED and that are modeled as explicit neonatal subcauses.
AFFECTED_CAUSES = list(CAUSES_OF_NEONATAL_MORTALITY)
# The lumped residual; per the plan the unaffected causes stay lumped as "other_causes".
UNAFFECTED_CAUSES = ["other_causes"]
ALL_NEONATAL_CAUSES = AFFECTED_CAUSES + UNAFFECTED_CAUSES

# Minimum sample sizes so proportion/RR checks fail meaningfully rather than on noise. If a
# category is too sparse at the run's population size, the check skips with a clear reason.
_MIN_BIRTHS_PER_CATEGORY = 200
_MIN_DEATHS_PER_CATEGORY = 10

# Plan-stated tolerance for expectation (5): RR within 10% of the GBD-derived ratio.
_RR_RELATIVE_TOLERANCE = 0.10

_EXPOSURE_COLUMNS = [
    COLUMNS.SEX_OF_CHILD,
    COLUMNS.CHILD_AGE,
    COLUMNS.PREGNANCY_OUTCOME,
    COLUMNS.CHILD_CAUSE_OF_DEATH,
    COLUMNS.GESTATIONAL_AGE_EXPOSURE,
    COLUMNS.BIRTH_WEIGHT_EXPOSURE,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _build_scenario_state(
    model_spec_path: Path,
    step_mapper: dict[str, int],
    scenario_name: str,
    step_name: str,
) -> InteractiveContext:
    """Build a sim under ``scenario_name`` and advance it to ``step_name``.

    Observers are dropped (we read the state table / pipelines directly) to keep the run
    light. Because both the baseline and intervention sims are built from the same spec they
    share the random seed, i.e. common random numbers.
    """
    if not _HAS_BUILD_SPEC:
        pytest.skip("build_model_specification unavailable; cannot override scenario")
    spec = build_model_specification(str(model_spec_path))
    try:
        del spec.configuration.observers
    except (AttributeError, KeyError):
        pass
    spec.configuration.intervention.scenario = scenario_name
    sim = InteractiveContext(spec)
    return get_interactive_context_state(sim, step_mapper, step_name)


@pytest.fixture(scope="module")
def require_affected_causes_artifact(artifact: Artifact) -> None:
    """Skip the whole module cleanly until the rebuilt artifact has the new key.

    The MIC-7320 artifact rebuild is deferred, so these checks are expected to skip now and
    run once the artifact is rebuilt.
    """
    try:
        keys = artifact.keys
    except Exception as exc:  # noqa: BLE001 - any failure to open the artifact => skip
        pytest.skip(f"artifact unavailable: {exc}")
    if AFFECTED_CAUSES_MORTALITY_RISK_KEY not in keys:
        pytest.skip(
            "artifact missing affected-causes mortality-risk key "
            f"'{AFFECTED_CAUSES_MORTALITY_RISK_KEY}'; rebuild the artifact for MIC-7320"
        )


@pytest.fixture(scope="module")
def baseline_early_neonatal_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    return _build_scenario_state(
        model_spec_path,
        sim_state_step_mapper,
        INTERVENTION_SCENARIOS.BASELINE.name,
        SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
    )


@pytest.fixture(scope="module")
def baseline_late_neonatal_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    return _build_scenario_state(
        model_spec_path,
        sim_state_step_mapper,
        INTERVENTION_SCENARIOS.BASELINE.name,
        SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
    )


@pytest.fixture(scope="module")
def intervention_early_neonatal_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    return _build_scenario_state(
        model_spec_path,
        sim_state_step_mapper,
        INTERVENTION_SCENARIO_NAME,
        SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
    )


@pytest.fixture(scope="module")
def intervention_late_neonatal_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    return _build_scenario_state(
        model_spec_path,
        sim_state_step_mapper,
        INTERVENTION_SCENARIO_NAME,
        SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draw(state: InteractiveContext) -> str:
    return f"draw_{state.model_specification.configuration.input_data.input_draw_number}"


def _category_series(state: InteractiveContext, sex: str) -> pd.Series:
    """Per-simulant LBWSG exposure category (post-intervention exposure) for one child sex."""
    pop = state.get_population(_EXPOSURE_COLUMNS)
    sim_exposure = get_simulation_exposure_categories(pop, state)
    sim_exposure = sim_exposure.loc[sim_exposure[COLUMNS.SEX_OF_CHILD] == sex]
    return sim_exposure["category"]


def _enn_deaths_and_births_by_category(
    state: InteractiveContext, sex: str, causes: list[str]
) -> pd.DataFrame:
    """Early-neonatal deaths (for ``causes``) and live births, tallied per LBWSG category."""
    pop = state.get_population(_EXPOSURE_COLUMNS)
    category = _category_series(state, sex)
    death_idx, birth_idx = get_births_and_deaths_idx(
        pop, sex, SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY, causes
    )
    death_idx = death_idx.intersection(category.index)
    birth_idx = birth_idx.intersection(category.index)
    tally = pd.DataFrame(
        {
            "deaths": category.loc[death_idx].value_counts(),
            "births": category.loc[birth_idx].value_counts(),
        }
    ).fillna(0.0)
    tally["rate"] = tally["deaths"] / tally["births"]
    return tally


def _combined_neonatal_counts(
    early_state: InteractiveContext,
    late_state: InteractiveContext,
    sex: str,
    causes: list[str],
) -> tuple[int, int]:
    """Combined early+late neonatal death count and live-birth denominator for ``causes``.

    Mirrors tests/test_mortality.py::test_neonatal_acmr: late-neonatal births have the
    early-neonatal deaths removed so no death is counted against two denominators.
    """
    enn = early_state.get_population(_EXPOSURE_COLUMNS)
    lnn = late_state.get_population(_EXPOSURE_COLUMNS)
    enn_death_idx, enn_birth_idx = get_births_and_deaths_idx(
        enn, sex, SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY, causes
    )
    lnn_death_idx, lnn_birth_idx = get_births_and_deaths_idx(
        lnn, sex, SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY, causes
    )
    lnn_birth_idx = lnn_birth_idx.difference(enn_death_idx)
    deaths = len(enn_death_idx) + len(lnn_death_idx)
    births = len(enn_birth_idx) + len(lnn_birth_idx)
    return deaths, births


def _artifact_enn_rr_by_category(artifact: Artifact, draw: str, sex: str) -> pd.Series:
    """Artifact early-neonatal LBWSG relative risk per category, for one child sex."""
    rr = artifact.load(LBWSG.RELATIVE_RISK)[draw].reset_index()
    sex_col = COLUMNS.SEX_OF_CHILD if COLUMNS.SEX_OF_CHILD in rr.columns else "sex"
    rr = rr[rr[sex_col] == sex]
    if "age_start" in rr.columns:
        # early neonatal age group starts at 0
        rr = rr[np.isclose(rr["age_start"], 0.0)]
    return rr.groupby("parameter")[draw].mean()


def _select_high_and_reference_categories(
    rr_by_category: pd.Series, tally: pd.DataFrame
) -> tuple[str, str]:
    """Pick a high-risk and a reference (RR nearest 1) category, both adequately sampled."""
    well_sampled = tally.index[
        (tally["births"] >= _MIN_BIRTHS_PER_CATEGORY)
        & (tally["deaths"] >= _MIN_DEATHS_PER_CATEGORY)
    ]
    candidates = rr_by_category.loc[rr_by_category.index.isin(well_sampled)]
    if len(candidates) < 2:
        pytest.skip(
            "fewer than two adequately-sampled LBWSG categories at this population size; "
            "increase population_size to run the RR / by-category checks"
        )
    high_cat = candidates.idxmax()
    reference_cat = (candidates - 1.0).abs().idxmin()
    if high_cat == reference_cat:
        pytest.skip("high-risk and reference LBWSG categories coincide; cannot form a ratio")
    return high_cat, reference_cat


# ---------------------------------------------------------------------------
# Expectation (1): affected-cause mortality varies by LBWSG category
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sex", ["Male", "Female"])
def test_affected_cause_mortality_varies_by_lbwsg_category(
    sex: str,
    require_affected_causes_artifact: None,
    baseline_early_neonatal_state: InteractiveContext,
    artifact: Artifact,
) -> None:
    """AFFECTED (preterm/sepsis/encephalopathy) neonatal mortality is higher in the
    high-LBWSG-risk category than in the reference category."""
    tally = _enn_deaths_and_births_by_category(
        baseline_early_neonatal_state, sex, AFFECTED_CAUSES
    )
    rr_by_category = _artifact_enn_rr_by_category(
        artifact, _draw(baseline_early_neonatal_state), sex
    )
    high_cat, reference_cat = _select_high_and_reference_categories(rr_by_category, tally)

    assert tally.loc[high_cat, "rate"] > tally.loc[reference_cat, "rate"], (
        f"[{sex}] affected-cause mortality did not increase with LBWSG risk: "
        f"{high_cat}={tally.loc[high_cat, 'rate']:.4g} vs "
        f"{reference_cat}={tally.loc[reference_cat, 'rate']:.4g}"
    )


# ---------------------------------------------------------------------------
# Expectation (2): unaffected ("other_causes") mortality varies by LBWSG category
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sex", ["Male", "Female"])
def test_unaffected_cause_mortality_varies_by_lbwsg_category(
    sex: str,
    require_affected_causes_artifact: None,
    baseline_early_neonatal_state: InteractiveContext,
    artifact: Artifact,
) -> None:
    """UNAFFECTED ("other_causes") neonatal mortality also varies across LBWSG categories
    (higher in the high-risk category than the reference)."""
    affected_tally = _enn_deaths_and_births_by_category(
        baseline_early_neonatal_state, sex, AFFECTED_CAUSES
    )
    rr_by_category = _artifact_enn_rr_by_category(
        artifact, _draw(baseline_early_neonatal_state), sex
    )
    # Reuse the affected tally to choose adequately-sampled high/reference categories.
    high_cat, reference_cat = _select_high_and_reference_categories(
        rr_by_category, affected_tally
    )
    other_tally = _enn_deaths_and_births_by_category(
        baseline_early_neonatal_state, sex, UNAFFECTED_CAUSES
    )
    if (
        other_tally.loc[reference_cat, "deaths"] < _MIN_DEATHS_PER_CATEGORY
        or other_tally.loc[high_cat, "deaths"] < _MIN_DEATHS_PER_CATEGORY
    ):
        pytest.skip("too few other-cause deaths per category at this population size")

    assert other_tally.loc[high_cat, "rate"] > other_tally.loc[reference_cat, "rate"], (
        f"[{sex}] other-cause mortality did not increase with LBWSG risk: "
        f"{high_cat}={other_tally.loc[high_cat, 'rate']:.4g} vs "
        f"{reference_cat}={other_tally.loc[reference_cat, 'rate']:.4g}"
    )


# ---------------------------------------------------------------------------
# Expectations (3) & (4): scenario response of unaffected vs affected mortality
# ---------------------------------------------------------------------------
def test_unaffected_mortality_invariant_to_intervention_coverage(
    require_affected_causes_artifact: None,
    baseline_early_neonatal_state: InteractiveContext,
    baseline_late_neonatal_state: InteractiveContext,
    intervention_early_neonatal_state: InteractiveContext,
    intervention_late_neonatal_state: InteractiveContext,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """UNAFFECTED ("other_causes") neonatal mortality is (near-)invariant to added
    LBWSG-affecting intervention coverage.

    Encoded two ways (see module docstring caveat about the other_causes mixture):
      * the intervention other-cause rate is statistically consistent with the baseline rate
        (FuzzyChecker), and
      * other-cause mortality changes far less, in relative terms, than affected-cause
        mortality (verified jointly with the affected-cause test below).
    """
    base_deaths, base_births = _combined_neonatal_counts(
        baseline_early_neonatal_state, baseline_late_neonatal_state, "Male", UNAFFECTED_CAUSES
    )
    f_deaths, f_births = _combined_neonatal_counts(
        baseline_early_neonatal_state,
        baseline_late_neonatal_state,
        "Female",
        UNAFFECTED_CAUSES,
    )
    base_deaths += f_deaths
    base_births += f_births

    int_deaths, int_births = _combined_neonatal_counts(
        intervention_early_neonatal_state,
        intervention_late_neonatal_state,
        "Male",
        UNAFFECTED_CAUSES,
    )
    fi_deaths, fi_births = _combined_neonatal_counts(
        intervention_early_neonatal_state,
        intervention_late_neonatal_state,
        "Female",
        UNAFFECTED_CAUSES,
    )
    int_deaths += fi_deaths
    int_births += fi_births

    baseline_rate = base_deaths / base_births
    # The intervention other-cause rate should be consistent with the baseline rate.
    fuzzy_checker.fuzzy_assert_proportion(
        int_deaths,
        int_births,
        baseline_rate,
        name="unaffected_other_causes_rate_invariant_to_intervention_coverage",
    )


def test_affected_mortality_changes_with_intervention_coverage(
    require_affected_causes_artifact: None,
    baseline_early_neonatal_state: InteractiveContext,
    baseline_late_neonatal_state: InteractiveContext,
    intervention_early_neonatal_state: InteractiveContext,
    intervention_late_neonatal_state: InteractiveContext,
) -> None:
    """AFFECTED-cause neonatal mortality changes under added LBWSG-affecting coverage, and
    changes MORE (in relative terms) than the unaffected causes.

    IFA/MMS scale-up raises birth weight and gestational age, moving simulants into
    lower-LBWSG-risk categories, so affected-cause mortality should fall.
    """

    def combined_both_sexes(early, late, causes):
        d0, b0 = _combined_neonatal_counts(early, late, "Male", causes)
        d1, b1 = _combined_neonatal_counts(early, late, "Female", causes)
        return d0 + d1, b0 + b1

    base_aff_d, base_aff_b = combined_both_sexes(
        baseline_early_neonatal_state, baseline_late_neonatal_state, AFFECTED_CAUSES
    )
    int_aff_d, int_aff_b = combined_both_sexes(
        intervention_early_neonatal_state, intervention_late_neonatal_state, AFFECTED_CAUSES
    )
    base_oth_d, base_oth_b = combined_both_sexes(
        baseline_early_neonatal_state, baseline_late_neonatal_state, UNAFFECTED_CAUSES
    )
    int_oth_d, int_oth_b = combined_both_sexes(
        intervention_early_neonatal_state, intervention_late_neonatal_state, UNAFFECTED_CAUSES
    )

    base_aff_rate = base_aff_d / base_aff_b
    int_aff_rate = int_aff_d / int_aff_b
    base_oth_rate = base_oth_d / base_oth_b
    int_oth_rate = int_oth_d / int_oth_b

    # (4) affected-cause mortality responds -- MMS/IFA is protective, so it falls.
    assert int_aff_rate < base_aff_rate, (
        "affected-cause neonatal mortality did not fall under added LBWSG-affecting "
        f"coverage: baseline={base_aff_rate:.4g}, {INTERVENTION_SCENARIO_NAME}={int_aff_rate:.4g}"
    )

    # (3) reinforced: affected causes change proportionally more than unaffected causes.
    affected_relative_change = abs(int_aff_rate - base_aff_rate) / base_aff_rate
    unaffected_relative_change = abs(int_oth_rate - base_oth_rate) / base_oth_rate
    assert affected_relative_change > unaffected_relative_change, (
        "affected-cause mortality should respond to LBWSG-affecting coverage more than "
        f"unaffected: affected rel. change={affected_relative_change:.4g}, "
        f"unaffected rel. change={unaffected_relative_change:.4g}"
    )


# ---------------------------------------------------------------------------
# Expectation (5): RR of neonatal death within 10% of the GBD-derived ratio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sex", ["Male", "Female"])
def test_relative_risk_of_neonatal_death_matches_gbd_within_10pct(
    sex: str,
    require_affected_causes_artifact: None,
    baseline_early_neonatal_state: InteractiveContext,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """In the baseline scenario the observed all-cause early-neonatal death rate ratio between
    a high-LBWSG-risk category and the reference category is within 10% of the GBD-derived
    ratio (the artifact LBWSG relative risk).

    This also guards the plan's algebraic identity: in the baseline scenario the decomposed
    ACMRisk must collapse to the original ``acmr*(1-PAF)*RR`` per category, so the observed
    all-cause RR by category should still track the artifact LBWSG RR.
    """
    draw = _draw(baseline_early_neonatal_state)
    all_cause_tally = _enn_deaths_and_births_by_category(
        baseline_early_neonatal_state, sex, ALL_NEONATAL_CAUSES
    )
    rr_by_category = _artifact_enn_rr_by_category(artifact, draw, sex)
    high_cat, reference_cat = _select_high_and_reference_categories(
        rr_by_category, all_cause_tally
    )

    observed_ratio = (
        all_cause_tally.loc[high_cat, "rate"] / all_cause_tally.loc[reference_cat, "rate"]
    )
    expected_ratio = rr_by_category.loc[high_cat] / rr_by_category.loc[reference_cat]

    # Record the underlying high-category proportion for diagnostics: expected proportion is
    # the observed reference-category rate scaled by the artifact RR ratio.
    fuzzy_checker.fuzzy_assert_proportion(
        int(all_cause_tally.loc[high_cat, "deaths"]),
        int(all_cause_tally.loc[high_cat, "births"]),
        all_cause_tally.loc[reference_cat, "rate"] * expected_ratio,
        name=f"{sex}_neonatal_death_rr_{high_cat}_vs_{reference_cat}",
    )

    assert np.isclose(observed_ratio, expected_ratio, rtol=_RR_RELATIVE_TOLERANCE), (
        f"[{sex}] observed neonatal-death RR {observed_ratio:.4g} "
        f"({high_cat} vs {reference_cat}) not within "
        f"{_RR_RELATIVE_TOLERANCE:.0%} of GBD-derived RR {expected_ratio:.4g}"
    )
