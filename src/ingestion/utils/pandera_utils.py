from typing import List

from pandera.pyspark import DataFrameModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def is_continuous_number(
    numbers: int, acceptble_max_values: List[int]
) -> bool:
    max_indexes = len(numbers)
    validated = False
    for index in range(0, max_indexes):
        prev = None if index == 0 else index - 1
        current = index
        next = None if (current == (max_indexes - 1)) else index + 1
        if prev and next:
            if (numbers[next] - numbers[current]) != 1:
                if (
                    numbers[next] != 1
                    or numbers[current] not in acceptble_max_values
                ):
                    validated = False
        validated = True
    return validated


def cast_table(
    df: DataFrame, pandera_dataframe_model: DataFrameModel
) -> DataFrame:
    schema = pandera_dataframe_model.to_schema()
    return df.select(
        *[
            col(schema_col.name).cast(schema_col.dtype.type)
            for schema_col in schema.columns.values()
        ]
    )
