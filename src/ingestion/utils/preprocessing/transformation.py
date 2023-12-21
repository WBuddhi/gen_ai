import re
from functools import reduce
from typing import List, Optional, Tuple

import pyspark.sql.functions as sf
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import collect_list, regexp_replace


class ColumnException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def replace_column_names(
    df: DataFrame, column_pair: List[Tuple[str, str]]
) -> DataFrame:
    output_df = df
    for original_name, replace_name in column_pair:
        if original_name not in df.columns:
            raise ColumnException(f"Column {original_name} not in DataFrame")
        output_df = output_df.withColumnRenamed(original_name, replace_name)
    return output_df


def create_column_pair_list(
    df: DataFrame, col_left: str, col_right: str
) -> DataFrame:
    left = df.select(collect_list(col_left)).first()[-1]
    right = df.select(collect_list(col_right)).first()[-1]
    if len(set(left)) == len(set(right)):
        column_pair = list(zip(left, right))
    else:
        raise ColumnException("Duplicate value in pair")
    return column_pair


def clean_column_names(df: DataFrame) -> DataFrame:
    output_df = df
    for original_column_name in output_df.columns:
        new_column_word_list = [
            word
            for word in (re.split(r"[^A-Za-z0-9]", original_column_name))
            if word != ""
        ]
        new_column_name = "_".join(new_column_word_list)
        output_df = output_df.withColumnRenamed(
            original_column_name, new_column_name.lower()
        )
    return output_df


def regex_replace_column(
    df, original_column, regex_replacement_pair
) -> DataFrame:
    df_with_regex_replace = df.withColumn(
        original_column,
        regexp_replace(original_column, *regex_replacement_pair),
    )
    return df_with_regex_replace


def join_table(
    left_table: DataFrame,
    right_table: DataFrame,
    join_pair: Tuple[str, str],
    how: str,
    drop_joined_column: bool = True,
    other_drop_columns: List[str] = None,
) -> DataFrame:
    left_table = left_table.withColumnRenamed(
        join_pair[0], left_name := f"left_{join_pair[0]}"
    )
    right_table = right_table.withColumnRenamed(
        join_pair[1], right_name := f"right_{join_pair[1]}"
    )
    joined_df = left_table.join(
        right_table,
        on=(left_table[left_name] == right_table[right_name]),
        how=how,
    )

    drop_columns = [] if other_drop_columns is None else other_drop_columns
    if drop_joined_column:
        drop_columns.extend([left_name, right_name])
    else:
        joined_df = joined_df.withColumnRenamed(left_name, join_pair[0])
        joined_df = joined_df.withColumnRenamed(right_name, join_pair[1])
    joined_df = joined_df.drop(*drop_columns) if drop_columns else joined_df

    return joined_df


def apply_prefix(df: DataFrame, prefix: str):
    for column in df.columns:
        df = df.withColumnRenamed(column, f"{prefix}_{column}")
    return df


def case_when_reduction(
    whens: List[Tuple[Column, Column]], otherwise: Optional[Column] = None
) -> Column:
    """Reduce case when statement

    Function that concatenates when statements that can be used for spark withColumn functions:
        Changes:
        >>> df.withColumn(
                "column",
                when(<expr>,<val>)
                .when(<expr>,<val>)
                .when(<expr>,<val>)
                .otherwise(<val>)
            )
        to:
        >>> df.withColumn(
            "column",
            case_when_reduction(
                whens = [
                    (<expr>,<val>),
                    (<expr>,<val>),
                    (<expr>,<val>)
                ],
                otherwise = <val>
            )
        )


    Args:
        whens (List[Tuple[Column, Column]]): list of when clauses.
        otherwise (Optional[Column], optional): optional otherwise statement. Defaults to None.

    Returns:
        Column: The computed when expression.

    Examples:
        >>> df = spark.createDataFrame(
                [(1, "mer", 3), (4, "mei", 6)], (7, "jun", 9)],
                schema=st.StructType(
                    [
                        st.StructField("A", st.StringType()),
                        st.StructField("B", st.StringType()),
                        st.StructField("C", st.StringType()),
                    ]
                ),
            )
        >>> df.show()
        +---+---------+---+
        |  A|    B    |  C|
        +---+---------+---+
        |  1|  "mer"  |  3|
        +---+---------+---+
        |  4|  "mei"  |  6|
        +---+---------+---+
        |  7|  "jun"  |  9|
        +---+---------+---+
        >>> df.withColumn(
                "B",
                case_when_reduction(
                    whens = [
                        (col("B").contains("mer"), lit("mar")),
                        (col("B").contains("mei"), lit("may"))
                    ],
                    otherwise = lit("B")
                )
            ).display()
        +---+---------+---+
        |  A|    B    |  C|
        +---+---------+---+
        |  1|  "mar"  |  3|
        +---+---------+---+
        |  4|  "may"  |  6|
        +---+---------+---+
        |  7|  "jun"  |  9|
        +---+---------+---+

    Raises:
        ColumnException: Raises ColumnException for incorrectly formed when clauses.
        error_info: Raises Exception for any other errors.
    """
    try:
        for item in whens:
            if type(item) != "<class 'tuple'>" and len(item) != 2:
                raise ColumnException(
                    "List should be a list of tuple pairs in the form [(,)]"
                )
        ret_col = reduce(lambda prev, kv: prev.when(kv[0], kv[1]), whens, sf)
        if otherwise is not None:
            ret_col = ret_col.otherwise(otherwise)
        return ret_col
    except Exception as error_info:
        raise error_info
