import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.time import DateTimeClock
from vivarium.framework.time import TimeInterface as TimeInterface_

from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES


class TimeInterface(TimeInterface_):
    def simulation_event_name(self):
        return lambda: self._manager.step_name


class EventClock(DateTimeClock):
    """A event driven clock that uses functionality of DateTimeClock."""

    @property
    def name(self):
        return "event_clock"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        # TODO: This means currently we need to configure the simulation to only have 4 steps
        self.simulation_events = (
            SIMULATION_EVENT_NAMES.INITIALIZATION,
            SIMULATION_EVENT_NAMES.PREGNANCY,
            SIMULATION_EVENT_NAMES.INTRAPARTRUM,
            SIMULATION_EVENT_NAMES.NEONATAL,
        )
        self.step_index = 0
        self.step_name = self.simulation_events[self.step_index]

    def step_forward(self, index: pd.Index) -> None:
        super().step_forward(index)
        if self.step_index < len(self.simulation_events) - 1:
            self.step_index += 1
