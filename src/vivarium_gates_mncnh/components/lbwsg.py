from vivarium.framework.engine import Builder
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRiskEffect as LBWSGRiskEffect_,
)
from vivarium_public_health.utilities import get_lookup_columns


class LBWSGRiskEffect(LBWSGRiskEffect_):
    """Subclass of LBWSGRiskEffect to expose the PAF pipeline to be accessable by other components."""

    def setup(self, builder: Builder) -> None:

        super().setup(builder)
        self.paf = builder.value.register_value_producer(
            "paf",
            source=self.lookup_tables["population_attributable_fraction"],
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["population_attributable_fraction"]]
            ),
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_paf_pipeline_name,
            modifier=self.paf,
            component=self,
        )
