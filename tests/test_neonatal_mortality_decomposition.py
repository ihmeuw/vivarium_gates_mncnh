"""White-box unit tests for the decomposed neonatal ACMRisk (MIC-7320).

These tests exercise ``NeonatalMortality.get_acmr_pipeline`` in isolation. They do NOT
build an ``InteractiveContext`` or load the artifact: the component is constructed without
running ``setup`` (via ``object.__new__``) and its three data dependencies are stubbed with
constant callables / a fake ``population_view`` so the algebra of the affected/unaffected
decomposition can be asserted deterministically.

The decomposition under test (see get_acmr_pipeline):

    affected   * (1 - PAF) * scenario_rr
  + unaffected * (1 - PAF) * baseline_rr

where ``unaffected = (acmr - affected).clip(lower=0)``.
"""

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_mncnh.components.mortality import NeonatalMortality
from vivarium_gates_mncnh.constants.data_values import PIPELINES


class _FakePopulationView:
    """Minimal population_view stub returning a constant Series per pipeline name."""

    def __init__(self, values: dict[str, float]) -> None:
        self._values = values

    def get(self, index: pd.Index, name: str) -> pd.Series:
        if name not in self._values:
            raise KeyError(f"unexpected pipeline requested: {name!r}")
        return pd.Series(self._values[name], index=index, name=name)


def _make_component(
    *,
    acmr: float,
    affected: float,
    paf: float,
    scenario_rr: float,
    baseline_rr: float,
) -> NeonatalMortality:
    """Build a NeonatalMortality with its data dependencies stubbed as constants.

    ``setup`` is bypassed so no builder/artifact is needed; only the three attributes that
    ``get_acmr_pipeline`` reads are populated.
    """
    component = object.__new__(NeonatalMortality)
    component.all_cause_mortality_risk = lambda index: pd.Series(acmr, index=index)
    component.affected_causes_mortality_risk = lambda index: pd.Series(affected, index=index)
    # ``population_view`` is a read-only property backed by ``_population_view``; set the
    # backing attribute directly since ``setup`` (which normally assigns it) is bypassed.
    component._population_view = _FakePopulationView(
        {
            PIPELINES.ACMR_PAF: paf,
            PIPELINES.ACMR_RR: scenario_rr,
            PIPELINES.ACMR_BASELINE_RR: baseline_rr,
        }
    )
    return component


_INDEX = pd.RangeIndex(5)


# ---------------------------------------------------------------------------
# (M) Deterministic and degenerate cases
# ---------------------------------------------------------------------------
def test_deterministic_decomposition() -> None:
    """acmr=1, affected=0.5, paf=0, scenario_rr=2, baseline_rr=1 -> 0.5*2 + 0.5*1 = 1.5."""
    component = _make_component(
        acmr=1.0, affected=0.5, paf=0.0, scenario_rr=2.0, baseline_rr=1.0
    )
    result = component.get_acmr_pipeline(_INDEX)
    assert np.allclose(result.values, 1.5)


def test_affected_equals_acmr_independent_of_baseline_rr() -> None:
    """When affected == acmr the unaffected term is 0, so baseline_rr is irrelevant."""
    kwargs = dict(acmr=0.8, affected=0.8, paf=0.1, scenario_rr=3.0)
    low = _make_component(baseline_rr=1.0, **kwargs).get_acmr_pipeline(_INDEX)
    high = _make_component(baseline_rr=7.0, **kwargs).get_acmr_pipeline(_INDEX)

    expected = 0.8 * (1 - 0.1) * 3.0  # affected * (1 - paf) * scenario_rr
    assert np.allclose(low.values, expected)
    assert np.allclose(low.values, high.values)


def test_affected_zero_independent_of_scenario_rr() -> None:
    """When affected == 0 the affected term is 0, so scenario_rr is irrelevant."""
    kwargs = dict(acmr=0.8, affected=0.0, paf=0.1, baseline_rr=2.0)
    low = _make_component(scenario_rr=1.0, **kwargs).get_acmr_pipeline(_INDEX)
    high = _make_component(scenario_rr=9.0, **kwargs).get_acmr_pipeline(_INDEX)

    expected = 0.8 * (1 - 0.1) * 2.0  # acmr * (1 - paf) * baseline_rr  (unaffected == acmr)
    assert np.allclose(low.values, expected)
    assert np.allclose(low.values, high.values)


def test_affected_exceeds_acmr_is_clamped_and_nonnegative() -> None:
    """affected > acmr: unaffected clamps to 0, result == affected*(1-paf)*scenario_rr >= 0."""
    component = _make_component(
        acmr=1.0, affected=1.5, paf=0.2, scenario_rr=2.0, baseline_rr=5.0
    )
    result = component.get_acmr_pipeline(_INDEX)

    expected = 1.5 * (1 - 0.2) * 2.0  # unaffected clamped to 0, baseline_rr drops out
    assert np.allclose(result.values, expected)
    assert (result.values >= 0).all()


# ---------------------------------------------------------------------------
# (N) Identity collapse: scenario_rr == baseline_rr == R
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("paf", [0.0, 0.15, 0.5, 0.9])
@pytest.mark.parametrize("R", [0.5, 1.0, 2.5])
def test_identity_collapse_when_rrs_equal(paf: float, R: float) -> None:
    """With scenario_rr == baseline_rr == R (and affected <= acmr), the decomposition
    collapses exactly to acmr*(1-paf)*R (to machine precision)."""
    acmr, affected = 1.0, 0.4  # affected <= acmr so affected + unaffected == acmr
    component = _make_component(
        acmr=acmr, affected=affected, paf=paf, scenario_rr=R, baseline_rr=R
    )
    result = component.get_acmr_pipeline(_INDEX)

    expected = acmr * (1 - paf) * R
    assert np.allclose(result.values, expected, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# (O) Target-modifier registration for all-cause LBWSGRiskEffect vs LBWSGPAFRiskEffect
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Verifying that the uniform ACMR target modifier is skipped for the main-sim "
        "all-cause LBWSGRiskEffect but registered for LBWSGPAFRiskEffect requires a "
        "fully set-up RiskEffect: LBWSGPAFRiskEffect.register_target_modifier delegates to "
        "RiskEffect.register_target_modifier, which reads attributes (target modifier "
        "source, relative-risk pipeline names, etc.) created only in setup() -- and setup "
        "needs the artifact. Covered instead by the InteractiveContext checks once the "
        "artifact is rebuilt."
    )
)
def test_acmr_target_modifier_registration() -> None:  # pragma: no cover - skipped
    raise NotImplementedError
