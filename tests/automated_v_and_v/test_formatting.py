import pandas as pd

from vivarium_gates_mncnh.validation.formatting import TotalLiveBirths


def test_births_formatter(get_births_observer_data: pd.DataFrame) -> None:
    """Test Births formatter."""

    formatter = TotalLiveBirths()

    assert formatter.measure == "live_births"
    assert formatter.raw_dataset_name == "births"
    assert formatter.unused_columns == ["measure", "entity_type", "entity", "sub_entity"]
    assert formatter.filters == {"sub_entity": ["total"]}

    expected_dataframe = pd.DataFrame(
        {
            "value": [70.0, 150.0] * 2,
        },
        index=pd.MultiIndex.from_tuples(
            [("Male", "A"), ("Male", "B"), ("Female", "A"), ("Female", "B")],
            names=["child_sex", "common_stratify_column"],
        ),
    )

    pd.testing.assert_frame_equal(
        formatter.format_dataset(get_births_observer_data), expected_dataframe
    )
