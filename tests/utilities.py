from vivarium import InteractiveContext


def get_interactive_context_state(
    sim: InteractiveContext, step_mapper: dict[str, int], step_name: str
) -> InteractiveContext:
    num_steps = step_mapper[step_name]
    sim.take_steps(num_steps)
    return sim
