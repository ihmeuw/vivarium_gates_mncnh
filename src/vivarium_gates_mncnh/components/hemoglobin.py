from __future__ import annotations

import numpy as np
import pandas as pd
import risk_distributions as rd
import scipy
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN, LookupTable
from vivarium.framework.population import SimulantData
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.data_transformations import pivot_categorical
from vivarium_public_health.risks.distributions import (
    EnsembleDistribution,
    MissingDataError,
    get_risk_distribution_parameter,
)
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect
from vivarium_public_health.utilities import EntityString

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    HEMORRHAGE_SEVERITY,
    PIPELINES,
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
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION:
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


class HemoglobinRiskEffect(NonLogLinearRiskEffect):
    """Make some small modifications to the NonLogLinearRiskEffect class to handle hemoglobin data.
    These are 1) define the RR for the minimum exposure to be the maximum rather than minimum value
    (higher hemoglobin is protective at this level of exposure) and 2) allow RRs to be below 1."""

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


class NonPregnantHemoglobinExposure(EnsembleDistribution):
    """Ensemble distribution for non-pregnant hemoglobin.

    Reuses the pregnant hemoglobin propensities (so a simulant draws the same
    quantile of its non-pregnant distribution as it did of its pregnant one) but
    swaps in the non-pregnant exposure mean. Standard deviation and ensemble
    weights are shared with the pregnant distribution.
    """

    def __init__(self) -> None:
        # Distinct risk name -> distinct ppf pipeline and lookup-table names, so
        # this does not collide with the pregnant hemoglobin distribution.
        super().__init__(EntityString("risk_factor.non_pregnant_hemoglobin"))
        # Drive the ppf with the propensities already initialized for the
        # pregnant hemoglobin Risk rather than registering new ones. These match
        # the columns created by Hemoglobin(Risk) ("risk_factor.hemoglobin") and
        # its EnsembleDistribution.
        self.risk_propensity = f"{data_keys.HEMOGLOBIN.name}.propensity"
        self.ensemble_propensity = (
            f"ensemble_propensity.risk_factor.{data_keys.HEMOGLOBIN.name}"
        )

    @property
    def name(self) -> str:
        return "non_pregnant_hemoglobin_exposure"

    def get_configuration(self, builder: Builder) -> None:
        # All data is loaded explicitly in get_distribution_definitions, so no
        # per-risk configuration tree is needed (and none exists for this risk).
        return None

    def setup(self, builder: Builder) -> None:
        distributions, weights, parameters = self.get_distribution_definitions(builder)
        self.distribution_weights_table = self.build_lookup_table(
            builder,
            "exposure_distribution_weights",
            data_source=weights,
            value_columns=distributions,
        )
        self.parameters = {
            parameter: self.build_lookup_table(
                builder,
                parameter,
                data_source=data.reset_index(),
                value_columns=rd.EnsembleDistribution.get_expected_parameters(parameter),
            )
            for parameter, data in parameters.items()
        }
        self.register_exposure_ppf_pipeline(builder)

    def get_distribution_definitions(
        self, builder: Builder
    ) -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
        # Non-pregnant mean, pregnant SD and pregnant ensemble weights, read
        # straight from the artifact (no per-risk configuration needed).
        exposure_data = builder.data.load(data_keys.HEMOGLOBIN.NON_PREGNANT_EXPOSURE)
        standard_deviation = builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION)
        raw_weights = builder.data.load(data_keys.HEMOGLOBIN.DISTRIBUTION_WEIGHTS)

        glnorm_mask = raw_weights["parameter"] == "glnorm"
        if np.any(raw_weights.loc[glnorm_mask, DEFAULT_VALUE_COLUMN]):
            raise NotImplementedError("glnorm distribution is not supported")
        raw_weights = raw_weights[~glnorm_mask]

        distributions = list(raw_weights["parameter"].unique())
        raw_weights = pivot_categorical(
            raw_weights, pivot_column="parameter", reset_index=False
        )

        weights, parameters = rd.EnsembleDistribution.get_parameters(
            raw_weights,
            mean=get_risk_distribution_parameter(exposure_data),
            sd=get_risk_distribution_parameter(standard_deviation),
        )
        return distributions, weights.reset_index(), parameters

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        tables = [self.distribution_weights_table, *self.parameters.values()]
        builder.value.register_attribute_producer(
            self.exposure_ppf_pipeline,
            source=self.exposure_ppf,
            required_resources=[*tables, self.risk_propensity, self.ensemble_propensity],
        )


class PostpartumHemoglobin(Component):
    """Model postpartum hemoglobin across two periods for surviving mothers.

    Applies the additive, negative hemorrhage hemoglobin shifts that stack across
    the antepartum and postpartum hemorrhage tracks:

    * 0-6 weeks: start from the end-of-pregnancy hemoglobin and apply the 0-6w shift.
    * 6 weeks-9 months: redraw hemoglobin from the non-pregnant distribution
      (same propensity as the pregnant draw) and apply the 6w-9m shift.

    Only mothers who survived labor (``is_alive``) and had hemorrhage receive a
    nonzero shift; everyone else keeps their value with a shift of zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.non_pregnant_distribution = NonPregnantHemoglobinExposure()

    @property
    def sub_components(self) -> list[Component]:
        return [self.non_pregnant_distribution]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

        self.shift_0_6w_tables = {
            COLUMNS.ANTEPARTUM_HEMORRHAGE: self.build_lookup_table(
                builder,
                "aph_shift_0_6w",
                data_source=data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.APH_SHIFT_0_6W,
            ),
            COLUMNS.POSTPARTUM_HEMORRHAGE: self.build_lookup_table(
                builder,
                "pph_shift_0_6w",
                data_source=data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.PPH_SHIFT_0_6W,
            ),
        }
        self.shift_6w_9m_tables = {
            COLUMNS.ANTEPARTUM_HEMORRHAGE: self.build_lookup_table(
                builder,
                "aph_shift_6w_9m",
                data_source=data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.APH_SHIFT_6W_9M,
            ),
            COLUMNS.POSTPARTUM_HEMORRHAGE: self.build_lookup_table(
                builder,
                "pph_shift_6w_9m",
                data_source=data_keys.HEMORRHAGE_HEMOGLOBIN_SHIFT.PPH_SHIFT_6W_9M,
            ),
        }

        self.non_pregnant_hemoglobin_pipeline_name = (
            PIPELINES.NON_PREGNANT_HEMOGLOBIN_EXPOSURE_PPF
        )

        builder.population.register_initializer(
            self.on_initialize_simulants,
            columns=[COLUMNS.POSTPARTUM_HEMOGLOBIN_EXPOSURE],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.initialize(
            pd.DataFrame(
                {COLUMNS.POSTPARTUM_HEMOGLOBIN_EXPOSURE: np.nan},
                index=pop_data.index,
            )
        )

    def on_time_step(self, event: Event) -> None:
        step = self._sim_step_name()
        if step == SIMULATION_EVENT_NAMES.POSTPARTUM_HEMOGLOBIN_6_WEEKS:
            self._apply_six_week_shift(event)
        elif step == SIMULATION_EVENT_NAMES.POSTPARTUM_HEMOGLOBIN_9_MONTHS:
            self._apply_nine_month_shift(event)

    def _apply_six_week_shift(self, event: Event) -> None:
        living_idx = self._living_mothers(event.index)
        pop = self.population_view.get(living_idx, [COLUMNS.HEMOGLOBIN_EXPOSURE])
        hemoglobin = pop[COLUMNS.HEMOGLOBIN_EXPOSURE] + self._total_shift(
            living_idx, self.shift_0_6w_tables
        )
        self._update_postpartum_hemoglobin(living_idx, hemoglobin)

    def _apply_nine_month_shift(self, event: Event) -> None:
        living_idx = self._living_mothers(event.index)
        non_pregnant_hemoglobin = self.population_view.get(
            living_idx, [self.non_pregnant_hemoglobin_pipeline_name]
        )[self.non_pregnant_hemoglobin_pipeline_name]
        hemoglobin = non_pregnant_hemoglobin + self._total_shift(
            living_idx, self.shift_6w_9m_tables
        )
        self._update_postpartum_hemoglobin(living_idx, hemoglobin)

    def _living_mothers(self, index: pd.Index) -> pd.Index:
        is_alive = self.population_view.get(index, [COLUMNS.MOTHER_ALIVE])
        return is_alive.loc[is_alive[COLUMNS.MOTHER_ALIVE]].index

    def _total_shift(
        self, index: pd.Index, shift_tables: dict[str, LookupTable]
    ) -> pd.Series:
        """Sum the per-track shifts, applying each only to that track's cases."""
        pop = self.population_view.get(
            index, [COLUMNS.ANTEPARTUM_HEMORRHAGE, COLUMNS.POSTPARTUM_HEMORRHAGE]
        )
        total = pd.Series(0.0, index=index)
        for column, table in shift_tables.items():
            had_hemorrhage = pop[column] != HEMORRHAGE_SEVERITY.NONE
            total = total + table(index).where(had_hemorrhage, 0.0)
        return total

    def _update_postpartum_hemoglobin(self, index: pd.Index, hemoglobin: pd.Series) -> None:
        self.population_view.update(
            pd.DataFrame(
                {COLUMNS.POSTPARTUM_HEMOGLOBIN_EXPOSURE: hemoglobin},
                index=index,
            )
        )
