import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX

from vivarium_gates_mncnh.validation.formatting import TotalLiveBirths


def test_births_formatter(get_births_observer_data: pd.DataFrame) -> None:
    """Test Births formatter."""

    formatter = TotalLiveBirths([])

    assert formatter.measure == "live_births"
    assert formatter.raw_dataset_name == "births"
    assert formatter.unused_columns == ["measure", "entity_type", "entity", "sub_entity"]
    assert formatter.filters == {"sub_entity": ["total"]}

    expected_dataframe = pd.DataFrame(
        {
            "value": [40.0, 50.0, 60.0, 70.0, 21.0, 27.0, 40.0, 50.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("Male", 0, 0),
                ("Male", 0, 1),
                ("Male", 1, 0),
                ("Male", 1, 1),
                ("Female", 0, 0),
                ("Female", 0, 1),
                ("Female", 1, 0),
                ("Female", 1, 1),
            ],
            names=["child_sex", DRAW_INDEX, SEED_INDEX],
        ),
    )

    pd.testing.assert_frame_equal(
        formatter.format_dataset(get_births_observer_data), expected_dataframe
    )
