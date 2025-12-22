import pandas as pd


def map_child_index_levels(data: pd.DataFrame) -> pd.DataFrame:
    """Maps index levels that start with 'child_' to remove the prefix."""
    data = data.rename_axis(
        index={
            name: name.replace("child_", "")
            for name in data.index.names
            if name and name.startswith("child_")
        }
    )
    # Handle artifact index level for sex
    if "sex_of_child" in data.index.names:
        data = data.rename_axis(index={"sex_of_child": "sex"})
    return data
