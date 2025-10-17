from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.distributions import MissingDataError
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import PIPELINES, SIMULATION_EVENT_NAMES


class Hemoglobin(Risk):
    @property
    def time_step_cleanup_priority(self) -> int:
        # oral iron intervention will have been assigned
        # when we update hemoglobin exposure in the state table
        return 7

    def __init__(self) -> None:
        super().__init__("risk_factor.hemoglobin")
        self.exposure_pipeline_name = f"raw_{self.risk.name}.exposure"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self._sim_step_name = builder.time.simulation_event_name()
        self.ifa_coverage = (
            builder.data.load(data_keys.IFA_SUPPLEMENTATION.COVERAGE)
            .query("parameter=='cat2'")
            .reset_index()
            .value[0]
        )
        self.ifa_effect_size = builder.data.load(
            data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE
        ).value[0]

        self.ifa_deleted_hemoglobin = builder.value.register_value_producer(
            PIPELINES.IFA_DELETED_HEMOGLOBIN_EXPOSURE,
            self.get_ifa_deleted_hemoglobin,
            component=self,
        )
        self.final_hemoglobin_exposure = builder.value.register_value_producer(
            PIPELINES.HEMOGLOBIN_EXPOSURE,
            source=self.ifa_deleted_hemoglobin,
            component=self,
        )
        builder.value.register_value_producer(
            PIPELINES.FIRST_ANC_HEMOGLOBIN_EXPOSURE,
            source=self.ifa_deleted_hemoglobin,
            component=self,
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables["ANC1"] = self.build_lookup_table(
            builder=builder,
            data_source=builder.data.load(data_keys.ANC.ANC1),
            value_columns=["value"],
        )

    def get_ifa_deleted_hemoglobin(self, index: pd.Index) -> pd.Series[float]:
        return self.exposure(index) - (
            self.ifa_effect_size * self.ifa_coverage * self.lookup_tables["ANC1"](index)
        )

    def on_time_step_cleanup(self, event: Event):
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.PREGNANCY:
            return
        exposure = self.final_hemoglobin_exposure(event.index)
        exposure.name = self.exposure_column_name
        self.population_view.update(exposure)


class HemoglobinRiskEffect(NonLogLinearRiskEffect):
    """Make some small modifications to the NonLogLinearRiskEffect class to handle hemoglobin data.
    These are 1) define the RR for the minimum exposure to be the maximum rather than minimum value
    (higher hemoglobin is protective at this level of exposure) and 2) allow RRs to be below 1."""

    def build_all_lookup_tables(self, builder: Builder) -> None:
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
            .apply(define_rr_intervals)
            .reset_index(level=-1, drop=True)
            .reset_index()
        )
        rr_data = rr_data.drop("parameter", axis=1)
        rr_data[f"{self.risk.name}_exposure_start"] = rr_data["left_exposure"]
        rr_data[f"{self.risk.name}_exposure_end"] = rr_data["right_exposure"]
        # build lookup table
        rr_value_cols = ["left_exposure", "left_rr", "right_exposure", "right_rr"]
        self.lookup_tables["relative_risk"] = self.build_lookup_table(
            builder, rr_data, rr_value_cols
        )

        paf_data = self.get_filtered_data(
            builder, self.configuration.data_sources.population_attributable_fraction
        )
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data
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
            .apply(get_rr_at_tmrel)
            .rename("rr_at_tmrel")
        )
        rr_data = original_rrs.merge(rrs_at_tmrel.reset_index())
        rr_data["value"] = rr_data["value"] / rr_data["rr_at_tmrel"]
        rr_data = rr_data.drop("rr_at_tmrel", axis=1)

        return rr_data
