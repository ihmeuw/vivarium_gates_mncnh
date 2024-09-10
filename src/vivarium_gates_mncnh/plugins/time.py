import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.time import DateTimeClock
from vivarium.framework.time import TimeInterface as TimeInterface_
from vivarium.framework.time import get_time_stamp

from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES


class TimeInterface(TimeInterface_):
    def simulation_event_name(self):
        return lambda: self._manager.step_name


class EventClock(DateTimeClock):
    """A event driven clock that uses functionality of DateTimeClock."""

    CONFIGURATION_DEFAULTS = {
        "time": {
            "start": {"year": None, "month": None, "day": None},
            "simulation_events": [],
        }
    }

    @property
    def name(self):
        return "event_clock"

    @property
    def step_name(self):
        return self.simulation_events[self.step_index]

    def setup(self, builder: Builder) -> None:
        super(DateTimeClock, self).setup(builder)
        self._minimum_step_size = pd.Timedelta(days=1)
        self._standard_step_size = pd.Timedelta(days=1)
        self._clock_step_size = self._minimum_step_size
        time = builder.configuration.time
        self.simulation_events = time.simulation_events
        self.step_index = 0
        self._clock_time = self.get_start_time(time)
        self._stop_time = self.get_end_time(time)

    def step_forward(self, index: pd.Index) -> None:
        super().step_forward(index)
        self.step_index += 1

    def step_backward(self) -> None:
        super().step_backward()
        self.step_index -= 1

    def get_start_time(self, time):
        # check that the start time is defined else raise an error
        if (
            time.start["year"] is None
            or time.start["month"] is None
            or time.start["day"] is None
        ):
            raise ValueError("Start time is not defined")
        return get_time_stamp(time.start)

    def get_end_time(self, time):
        # check that only one of the end time or number of steps is defined
        number_of_steps = len(time.simulation_events)
        if not time.simulation_events:
            raise ValueError("No simulation events defined in the configuration.")

        return get_time_stamp(time.start) + number_of_steps * self._clock_step_size
