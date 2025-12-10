import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX

from vivarium_gates_mncnh.validation.formatting import CauseDeaths, LiveBirths


def test_births_formatter(get_births_observer_data: pd.DataFrame) -> None:
    """Test Births formatter."""

    formatter = LiveBirths([])

    assert formatter.measure == "live_births"
    assert formatter.raw_dataset_name == "births"
    assert formatter.unused_columns == ["measure", "entity_type", "entity", "sub_entity"]
    assert formatter.filters == {"pregnancy_outcome": ["live_birth"]}

    expected_dataframe = pd.DataFrame(
        {
            "value": [
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                1.0,
                2.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("Male", "A", 0, 0),
                ("Male", "A", 0, 1),
                ("Male", "A", 1, 0),
                ("Male", "A", 1, 1),
                ("Male", "B", 0, 0),
                ("Male", "B", 0, 1),
                ("Male", "B", 1, 0),
                ("Male", "B", 1, 1),
                ("Female", "A", 0, 0),
                ("Female", "A", 0, 1),
                ("Female", "A", 1, 0),
                ("Female", "A", 1, 1),
                ("Female", "B", 0, 0),
                ("Female", "B", 0, 1),
                ("Female", "B", 1, 0),
                ("Female", "B", 1, 1),
            ],
            names=["child_sex", "common_stratify_column", DRAW_INDEX, SEED_INDEX],
        ),
    )

    pd.testing.assert_frame_equal(
        formatter.format_dataset(get_births_observer_data), expected_dataframe
    )


def test_deaths_formatter(get_deaths_observer_data: pd.DataFrame) -> None:
    """Test Deaths formatter."""

    cause = "neonatal_testing"
    formatter = CauseDeaths(cause)

    assert formatter.measure == cause
    assert formatter.raw_dataset_name == f"{cause}_death_counts"
    assert formatter.unused_columns == ["measure", "entity_type", "entity", "sub_entity"]
    assert formatter.filters == {"sub_entity": ["total"]}

    expected = get_deaths_observer_data.copy().droplevel(
        ["measure", "entity_type", "entity", "sub_entity"]
    )
    pd.testing.assert_frame_equal(
        formatter.format_dataset(get_deaths_observer_data), expected
    )
