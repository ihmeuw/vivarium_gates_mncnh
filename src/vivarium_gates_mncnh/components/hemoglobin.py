from __future__ import annotations

import numpy as np
import pandas as pd
import risk_distributions as rd
import scipy
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.data_transformations import pivot_categorical
from vivarium_public_health.risks.distributions import MissingDataError
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    HEMORRHAGE_CAUSES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import (
    clip_quantiles,
    get_risk_distribution_parameter,
)


class Hemoglobin(Risk):
    @property
    def time_step_priority(self) -> int:
        # update state table hemoglobin after oral iron
        return 9

    def __init__(self) -> None:
        super().__init__("risk_factor.hemoglobin")

    def setup(self, builder: Builder) -> None:
        super().setup(builder)

        self.anc1_table = self.build_lookup_table(
            builder,
            "ANC1",
            data_source=builder.data.load(data_keys.ANC.ANC1),
            value_columns="value",
        )

        self._sim_step_name = builder.time.simulation_event_name()
        self.ifa_coverage = (
            builder.data.load(data_keys.IFA_SUPPLEMENTATION.COVERAGE)
            .query("parameter=='cat2'")
            .reset_index()
            .value[0]
        )
        self.ifa_effect_size = (
            builder.data.load(data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE)
            .query("affected_target=='hemoglobin.exposure'")
            .reset_index()
            .value[0]
        )

        builder.population.register_initializer(
            initializer=self._initialize_hemoglobin_columns,
            columns=[
                COLUMNS.HEMOGLOBIN_EXPOSURE,
                COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE,
            ],
        )

        builder.value.register_attribute_modifier(
            self.exposure_name,
            modifier=self._adjust_exposure_for_ifa,
        )

        # Postpartum hemoglobin: load hemorrhage shift data and non-pregnant
        # exposure data, then register a pipeline modifier that applies
        # hemorrhage shifts at the 0-6w and 6w-9m postpartum events.
        self.pph_shift_0_6w = builder.data.load(
            data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.PPH_SHIFT_0_6W
        )["value"].item()
        self.aph_shift_0_6w = builder.data.load(
            data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.APH_SHIFT_0_6W
        )["value"].item()
        self.pph_shift_6w_9m = builder.data.load(
            data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.PPH_SHIFT_6W_9M
        )["value"].item()
        self.aph_shift_6w_9m = builder.data.load(
            data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.APH_SHIFT_6W_9M
        )["value"].item()
        self._build_non_pregnant_distribution(builder)

        builder.value.register_attribute_modifier(
            self.exposure_name,
            modifier=self._modify_exposure_for_postpartum,
        )

    def _build_non_pregnant_distribution(self, builder: Builder) -> None:
        """Build ensemble distribution machinery for non-pregnant hemoglobin sampling."""
        non_pregnant_mean = builder.data.load(data_keys.HEMOGLOBIN.NON_PREGNANT_EXPOSURE)
        sd_data = builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION)
        raw_weights = builder.data.load(data_keys.HEMOGLOBIN.DISTRIBUTION_WEIGHTS)

        # glnorm not currently supported by risk_distributions
        glnorm_mask = raw_weights["parameter"] == "glnorm"
        raw_weights = raw_weights[~glnorm_mask]
        distributions = list(raw_weights["parameter"].unique())
        raw_weights = pivot_categorical(
            raw_weights, pivot_column="parameter", reset_index=False
        )

        weights, parameters = rd.EnsembleDistribution.get_parameters(
            raw_weights,
            mean=get_risk_distribution_parameter(non_pregnant_mean),
            sd=get_risk_distribution_parameter(sd_data),
        )

        self._non_pregnant_weights_table = self.build_lookup_table(
            builder,
            "non_pregnant_weights",
            data_source=weights.reset_index(),
            value_columns=distributions,
        )
        self._non_pregnant_parameters = {
            param_name: self.build_lookup_table(
                builder,
                f"non_pregnant_{param_name}",
                data_source=param_data.reset_index(),
                value_columns=list(param_data.columns),
            )
            for param_name, param_data in parameters.items()
        }

        self._propensity_view = builder.population.get_view(
            [self.propensity_name, f"ensemble_propensity.{self.risk}"]
        )

    def _initialize_hemoglobin_columns(self, pop_data: SimulantData) -> None:
        pop = pd.DataFrame(
            {
                COLUMNS.HEMOGLOBIN_EXPOSURE: np.nan,
                COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE: np.nan,
            },
            index=pop_data.index,
        )
        self.population_view.initialize(pop)

    def _adjust_exposure_for_ifa(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        return exposure - (self.ifa_effect_size * self.ifa_coverage * self.anc1_table(index))

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        event_name = self._sim_step_name()
        if event_name not in (
            SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION,
            SIMULATION_EVENT_NAMES.EARLY_POSTPARTUM,
            SIMULATION_EVENT_NAMES.LATE_POSTPARTUM,
        ):
            return
        # At LATER_PREGNANCY_INTERVENTION: snapshot of pregnancy hemoglobin
        # (pre-partum, before any hemorrhage shifts).
        # At EARLY_POSTPARTUM (0-6w postpartum): hemoglobin after
        # antepartum/postpartum hemorrhage shifts have been applied.
        # At LATE_POSTPARTUM (6w-9m): hemoglobin drawn from
        # the non-pregnant distribution with hemorrhage shifts applied.
        exposure = self.population_view.get(event.index, self.exposure_name)
        exposure = exposure.rename(COLUMNS.HEMOGLOBIN_EXPOSURE)
        self.population_view.update(
            COLUMNS.HEMOGLOBIN_EXPOSURE,
            lambda _: exposure,
        )

    def on_time_step_cleanup(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC:
            return
        exposure = self.population_view.get(event.index, self.exposure_name)
        exposure = exposure.rename(COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE)
        self.population_view.update(
            COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE,
            lambda _: exposure,
        )

    ################################
    # Postpartum pipeline modifier #
    ################################

    def _modify_exposure_for_postpartum(
        self, index: pd.Index, exposure: pd.Series
    ) -> pd.Series:
        """Apply hemorrhage hemoglobin shifts at postpartum events.

        At ``early_postpartum`` (0-6 week postpartum): apply PPH and APH
        shifts to the existing pregnancy hemoglobin for hemorrhage cases.

        At ``late_postpartum`` (6w-9m): replace pregnancy
        hemoglobin with a draw from the non-pregnant distribution and apply
        PPH/APH shifts for hemorrhage cases.

        At all other events this is a no-op.
        """
        event_name = self._sim_step_name()

        if event_name == SIMULATION_EVENT_NAMES.EARLY_POSTPARTUM:
            return self._apply_0_6w_shifts(index, exposure)
        elif event_name == SIMULATION_EVENT_NAMES.LATE_POSTPARTUM:
            return self._apply_6w_9m_shifts(index, exposure)
        else:
            return exposure

    def _get_postpartum_pop_and_mask(self, index: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
        """Fetch postpartum population columns and compute the survived mask."""
        pop = self.population_view.get(
            index,
            [
                COLUMNS.PREGNANCY_OUTCOME,
                COLUMNS.POSTPARTUM_HEMORRHAGE,
                COLUMNS.ANTEPARTUM_HEMORRHAGE,
            ],
        )
        survived_mask = pop[COLUMNS.PREGNANCY_OUTCOME].isin(
            [PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME, PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME]
        )
        return pop, survived_mask

    def _apply_hemorrhage_shifts(
        self, hgb: pd.Series, pop: pd.DataFrame, pph_shift: float, aph_shift: float
    ) -> pd.Series:
        """Apply PPH and APH hemorrhage shifts to hemoglobin values.

        Shifts are applied additively; simulants with both conditions
        receive both shifts.
        """
        pph_mask = pop[COLUMNS.POSTPARTUM_HEMORRHAGE].fillna(False)
        if pph_mask.any():
            hgb.loc[pph_mask] += pph_shift

        aph_mask = pop[COLUMNS.ANTEPARTUM_HEMORRHAGE].fillna(False)
        if aph_mask.any():
            hgb.loc[aph_mask] += aph_shift

        return hgb.clip(lower=0)

    def _apply_0_6w_shifts(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        """Apply 0-6 week hemorrhage shifts to pregnancy hemoglobin."""
        pop, survived_mask = self._get_postpartum_pop_and_mask(index)

        if not survived_mask.any():
            return exposure

        survived_pop = pop.loc[survived_mask]
        result = exposure.copy()
        result.loc[survived_pop.index] = self._apply_hemorrhage_shifts(
            result.loc[survived_pop.index].copy(),
            survived_pop,
            self.pph_shift_0_6w,
            self.aph_shift_0_6w,
        )
        return result

    def _apply_6w_9m_shifts(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        """Replace pregnancy hemoglobin with non-pregnant values and apply 6w-9m shifts."""
        pop, survived_mask = self._get_postpartum_pop_and_mask(index)
        survived_pop = pop.loc[survived_mask]

        if survived_pop.empty:
            return exposure

        hgb = self._sample_non_pregnant_hemoglobin(survived_pop.index)
        hgb = self._apply_hemorrhage_shifts(
            hgb, survived_pop, self.pph_shift_6w_9m, self.aph_shift_6w_9m
        )

        result = exposure.copy()
        result.loc[survived_pop.index] = hgb
        return result

    def _sample_non_pregnant_hemoglobin(self, index: pd.Index) -> pd.Series:
        """Sample hemoglobin values from the non-pregnant ensemble distribution."""
        propensities = self._propensity_view.get(
            index, [self.propensity_name, f"ensemble_propensity.{self.risk}"]
        )
        quantiles = clip_quantiles(propensities[self.propensity_name])
        ensemble_propensities = propensities[f"ensemble_propensity.{self.risk}"]
        weights = self._non_pregnant_weights_table(index)
        parameters = {
            name: param(index) for name, param in self._non_pregnant_parameters.items()
        }
        result = rd.EnsembleDistribution(weights, parameters).ppf(
            quantiles, ensemble_propensities
        )
        result[result.isnull()] = 0
        return result


class HemoglobinRiskEffect(NonLogLinearRiskEffect):
    """Make some small modifications to the NonLogLinearRiskEffect class to handle hemoglobin data.
    These are 1) define the RR for the minimum exposure to be the maximum rather than minimum value
    (higher hemoglobin is protective at this level of exposure) and 2) allow RRs to be below 1."""

    # APH and PPH share the same RR/PAF data keyed as "maternal_hemorrhage" in the artifact
    HEMORRHAGE_ENTITY_REMAP = {cause: "maternal_hemorrhage" for cause in HEMORRHAGE_CAUSES}

    def get_filtered_data(
        self, builder: Builder, data_source: str | float | pd.DataFrame
    ) -> float | pd.DataFrame:
        """Load and filter RR/PAF data for this target, with hemorrhage remapping.

        APH and PPH are modeled as separate causes in the component configuration,
        but hemoglobin RR/PAF artifact data is keyed under the shared
        ``affected_entity == 'maternal_hemorrhage'``. For those two targets, we
        remap ``self.target.name`` to the shared artifact entity before filtering.

        When present, both ``affected_entity`` and ``affected_measure`` columns are
        used to select only rows relevant to this target, and then dropped so the
        returned frame matches the shape expected by downstream interpolation logic.
        """
        data = self.get_data(builder, data_source)

        if isinstance(data, pd.DataFrame):
            filter_entity = self.HEMORRHAGE_ENTITY_REMAP.get(
                self.target.name, self.target.name
            )
            correct_target_mask = pd.Series(True, index=data.index)
            columns_to_drop = []
            if "affected_entity" in data.columns:
                correct_target_mask &= data["affected_entity"] == filter_entity
                columns_to_drop.append("affected_entity")
            if "affected_measure" in data.columns:
                correct_target_mask &= data["affected_measure"] == self.target.measure
                columns_to_drop.append("affected_measure")
            data = data[correct_target_mask].drop(columns=columns_to_drop)
        return data

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        rr_data = self.load_relative_risk(builder)
        self.validate_rr_data(rr_data)

        def define_rr_intervals(df: pd.DataFrame) -> pd.DataFrame:
            # create new row for right-most exposure bin (RR is same as max RR)
            max_exposure_row = df.tail(1).copy()
            max_exposure_row["parameter"] = np.inf
            rr_data = pd.concat([df, max_exposure_row]).reset_index()

            rr_data["left_exposure"] = [0] + rr_data["parameter"][:-1].tolist()
            # use max rather than min value
            rr_data["left_rr"] = [rr_data["value"].max()] + rr_data["value"][:-1].tolist()
            rr_data["right_exposure"] = rr_data["parameter"]
            rr_data["right_rr"] = rr_data["value"]

            return rr_data[
                ["parameter", "left_exposure", "left_rr", "right_exposure", "right_rr"]
            ]

        # define exposure and rr interval columns
        demographic_cols = [
            col for col in rr_data.columns if col != "parameter" and col != "value"
        ]
        rr_data = (
            rr_data.groupby(demographic_cols)
            .apply(define_rr_intervals, include_groups=False)
            .reset_index(level=-1, drop=True)
            .reset_index()
        )
        rr_data = rr_data.drop("parameter", axis=1)
        rr_data[f"{self.risk.name}_exposure_for_non_loglinear_riskeffect_start"] = rr_data[
            "left_exposure"
        ]
        rr_data[f"{self.risk.name}_exposure_for_non_loglinear_riskeffect_end"] = rr_data[
            "right_exposure"
        ]
        # build lookup table
        rr_value_cols = ["left_exposure", "left_rr", "right_exposure", "right_rr"]
        return self.build_lookup_table(
            builder, "relative_risk", data_source=rr_data, value_columns=rr_value_cols
        )

    def load_relative_risk(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        if configuration is None:
            configuration = self.configuration

        # get TMREL
        tmred = builder.data.load(f"{self.risk}.tmred")
        if tmred["distribution"] == "uniform":
            draw = builder.configuration.input_data.input_draw_number
            rng = np.random.default_rng(builder.randomness.get_seed(self.name + str(draw)))
            self.tmrel = rng.uniform(tmred["min"], tmred["max"])
        elif tmred["distribution"] == "draws":  # currently only for iron deficiency
            raise MissingDataError(
                f"This data has draw-level TMRELs. You will need to contact the research team that models {self.risk.name} to get this data."
            )
        else:
            raise MissingDataError(f"No TMRED found in gbd_mapping for risk {self.risk.name}")

        # calculate RR at TMREL
        rr_source = configuration.data_sources.relative_risk
        original_rrs = self.get_filtered_data(builder, rr_source)

        self.validate_rr_data(original_rrs)

        demographic_cols = [
            col for col in original_rrs.columns if col != "parameter" and col != "value"
        ]

        def get_rr_at_tmrel(rr_data: pd.DataFrame) -> float:
            interpolated_rr_function = scipy.interpolate.interp1d(
                rr_data["parameter"],
                rr_data["value"],
                kind="linear",
                bounds_error=False,
                fill_value=(
                    rr_data["value"].min(),
                    rr_data["value"].max(),
                ),
            )
            rr_at_tmrel = interpolated_rr_function(self.tmrel).item()
            return rr_at_tmrel

        rrs_at_tmrel = (
            original_rrs.groupby(demographic_cols)
            .apply(get_rr_at_tmrel, include_groups=False)
            .rename("rr_at_tmrel")
        )
        rr_data = original_rrs.merge(rrs_at_tmrel.reset_index())
        rr_data["value"] = rr_data["value"] / rr_data["rr_at_tmrel"]
        rr_data = rr_data.drop("rr_at_tmrel", axis=1)

        return rr_data
