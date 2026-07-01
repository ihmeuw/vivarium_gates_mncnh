"""Lightweight simulation helpers with no vivarium_inputs dependency."""

from vivarium.engine import InteractiveContext
from vivarium.engine.framework.configuration import build_model_specification

from vivarium_gates_mncnh.constants.paths import MODEL_SPEC_PATH


def initialize_simulation(location: str, draw: int, population_size: int):
    """Build an InteractiveContext from the default model spec.

    Parameters
    ----------
    location
        Lower-case location name used to select the artifact file.
    draw
        Input draw number.
    population_size
        Simulation population size.
    """
    spec = build_model_specification(MODEL_SPEC_PATH)
    del spec.configuration.observers
    artifact_base = spec.configuration.input_data.artifact_path.rsplit("/", 1)[0] + "/"
    spec.configuration.input_data.artifact_path = artifact_base + location + ".hdf"
    spec.configuration.input_data.input_draw_number = draw
    spec.configuration.population.population_size = population_size
    return InteractiveContext(spec)
