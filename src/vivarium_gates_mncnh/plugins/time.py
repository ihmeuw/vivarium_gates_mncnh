import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.time import DateTimeClock
from vivarium.framework.time import TimeInterface as TimeInterface_


class TimeInterface(TimeInterface_):
    def step_size_name(self):
        return self._manager.step_name


class EventClock(DateTimeClock):
    """A event driven clock."""

    @property
    def name(self):
        return "event_clock"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.simulation_events = ("initialization", "pregnancy", "intrapartrum", "neonatal")
        self.step_index = 0
        self.step_name = self.simulation_events[self.step_index]

    def step_forward(self, index: pd.Index) -> None:
        super().step_forward(index)
        if self.step_index < len(self.simulation_events) - 1:
            self.step_index += 1
