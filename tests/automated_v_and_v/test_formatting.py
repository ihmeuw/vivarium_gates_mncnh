import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX

from vivarium_gates_mncnh.validation.formatting import (
    CauseDeaths,
    LiveBirths,
    map_child_index_levels,
)


def test_births_formatter(births_observer_data: pd.DataFrame) -> None:
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
                11.0,
                12.0,
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
            names=["sex", "common_stratify_column", DRAW_INDEX, SEED_INDEX],
        ),
    )

    pd.testing.assert_frame_equal(
        formatter.format_dataset(births_observer_data), expected_dataframe
    )


def test_deaths_formatter(deaths_observer_data: pd.DataFrame) -> None:
    """Test Deaths formatter."""

    cause = "neonatal_testing"
    formatter = CauseDeaths(cause)

    assert formatter.measure == cause
    assert formatter.raw_dataset_name == f"{cause}_death_counts"
    assert formatter.unused_columns == ["measure", "entity_type", "entity", "sub_entity"]
    assert formatter.filters == {"sub_entity": ["total"]}

    # This is implementing the format_dataset method
    expected = deaths_observer_data.copy().droplevel(
        ["measure", "entity_type", "entity", "sub_entity"]
    )
    expected = map_child_index_levels(expected)
    pd.testing.assert_frame_equal(formatter.format_dataset(deaths_observer_data), expected)


def test_child_data_formatter(
    deaths_observer_data: pd.DataFrame,
) -> None:
    """Test ChildDataFormatter removes 'child_' prefix from columns."""

    formatter = CauseDeaths("neonatal_testing")
    assert "child_age_group" in deaths_observer_data.index.names
    assert "child_sex" in deaths_observer_data.index.names

    formatted = formatter.format_dataset(deaths_observer_data)

    # Check that no columns start with 'child_'
    for level in formatted.index.names:
        assert not level.startswith("child_")
