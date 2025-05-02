import pandas as pd
from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import COLUMNS, PREGNANCY_OUTCOMES


def get_interactive_context_state(
    sim: InteractiveContext, step_mapper: dict[str, int], step_name: str
) -> InteractiveContext:
    num_steps = step_mapper[step_name]
    sim.take_steps(num_steps)
    return sim


def get_births_and_deaths_idx(
    pop: pd.DataFrame, child_sex: str, step_name: str, causes_of_death: list[str]
) -> tuple[pd.Index, pd.Index]:
    pop_filters = {
        "early_neonatal_mortality": "child_age > 1 / 365 and child_age < 7 / 365",
        "late_neonatal_mortality": "child_age > 7 / 365 and child_age < 28 / 365",
    }
    pop = pop.loc[pop[COLUMNS.SEX_OF_CHILD] == child_sex]
    pop = pop.query(pop_filters[step_name])
    live_birth_idx = pop.index[
        pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
    ]
    death_idx = pop.index[pop[COLUMNS.CHILD_CAUSE_OF_DEATH].isin(causes_of_death)]
    return death_idx, live_birth_idx
