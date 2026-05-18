from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.distributions import MissingDataError
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    HEMORRHAGE_CAUSES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
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
        self.non_pregnant_exposure_data = builder.data.load(
            data_keys.HEMOGLOBIN.NON_PREGNANT_EXPOSURE
        )

        builder.value.register_attribute_modifier(
            self.exposure_name,
            modifier=self._modify_exposure_for_postpartum,
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
        if self._sim_step_name() not in (
            SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION,
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.POSTPARTUM_HEMOGLOBIN_NINE_MONTH,
        ):
            return
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

    ##############################
    # Postpartum pipeline modifier
    ##############################

    def _modify_exposure_for_postpartum(
        self, index: pd.Index, exposure: pd.Series
    ) -> pd.Series:
        """Apply hemorrhage hemoglobin shifts at postpartum events.

        At ``early_neonatal_mortality`` (0-6 week postpartum): apply PPH and APH
        shifts to the existing pregnancy hemoglobin for hemorrhage cases.

        At ``postpartum_hemoglobin_nine_month`` (6w-9m): replace pregnancy
        hemoglobin with a draw from the non-pregnant distribution and apply
        PPH/APH shifts for hemorrhage cases.

        At all other events this is a no-op.
        """
        event_name = self._sim_step_name()

        if event_name == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY:
            return self._apply_0_6w_shifts(index, exposure)
        elif event_name == SIMULATION_EVENT_NAMES.POSTPARTUM_HEMOGLOBIN_NINE_MONTH:
            return self._apply_6w_9m_shifts(index, exposure)
        else:
            return exposure

    def _apply_0_6w_shifts(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        """Apply 0-6 week hemorrhage shifts to pregnancy hemoglobin."""
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

        if not survived_mask.any():
            return exposure

        result = exposure.copy()

        pph_mask = survived_mask & (pop[COLUMNS.POSTPARTUM_HEMORRHAGE] == True)
        if pph_mask.any():
            result.loc[pph_mask] = result.loc[pph_mask] + self.pph_shift_0_6w

        aph_mask = survived_mask & (pop[COLUMNS.ANTEPARTUM_HEMORRHAGE] == True)
        if aph_mask.any():
            result.loc[aph_mask] = result.loc[aph_mask] + self.aph_shift_0_6w

        return result.clip(lower=0)

    def _apply_6w_9m_shifts(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        """Replace pregnancy hemoglobin with non-pregnant values and apply 6w-9m shifts."""
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
        survived_idx = pop.loc[survived_mask].index

        if survived_idx.empty:
            return exposure

        hgb = self._draw_non_pregnant_hemoglobin(survived_idx)

        pph_mask = survived_mask & (pop[COLUMNS.POSTPARTUM_HEMORRHAGE] == True)
        hgb.loc[pph_mask] = hgb.loc[pph_mask] + self.pph_shift_6w_9m

        aph_mask = survived_mask & (pop[COLUMNS.ANTEPARTUM_HEMORRHAGE] == True)
        hgb.loc[aph_mask] = hgb.loc[aph_mask] + self.aph_shift_6w_9m

        hgb = hgb.clip(lower=0)

        result = exposure.copy()
        result.loc[survived_idx] = hgb
        return result

    def _draw_non_pregnant_hemoglobin(self, index: pd.Index) -> pd.Series:
        """Draw hemoglobin values from the non-pregnant distribution."""
        pop = self.population_view.get(index, ["age"])
        return self._lookup_by_age(pop["age"], self.non_pregnant_exposure_data)

    def _lookup_by_age(self, ages: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Look up values from age-binned data for each simulant's age."""
        result = pd.Series(np.nan, index=ages.index)

        if "age_start" in data.index.names:
            data_reset = data.reset_index()
        else:
            data_reset = data

        for _, row in data_reset.iterrows():
            age_start = row.get("age_start", 0)
            age_end = row.get("age_end", 200)
            mask = (ages >= age_start) & (ages < age_end)
            result.loc[mask] = row["value"]

        if result.isna().any():
            result = result.fillna(data_reset["value"].mean())

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
