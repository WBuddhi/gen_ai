from pyspark.sql import Column
from pyspark.sql.functions import col, lit, to_timestamp

from src.utils.preprocessing.transformation import (
    case_when_reduction,
    regexp_replace,
)


def format_time_string(pattern: str, replacement: str) -> Column:
    return to_timestamp(
        regexp_replace(
            regexp_replace(col("time").substr(5, 16), ",", ""),
            pattern,
            replacement,
        ),
        "dd-MMM-yyyy",
    )


def circana_time_transformation() -> Column:
    return case_when_reduction(
        whens=[
            (
                col("time").contains(lit("okt")),
                format_time_string(" okt ", "-oct-"),
            ),
            (
                col("time").contains(lit("mrt")),
                format_time_string(" mrt ", "-mar-"),
            ),
            (
                col("time").contains(lit("mei")),
                format_time_string(" mei ", "-may-"),
            ),
        ],
        otherwise=format_time_string(" ", "-"),
    )
