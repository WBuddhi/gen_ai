import os
from datetime import datetime
from typing import List, Union

from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from databricks.sdk import WorkspaceClient
from pyspark.sql.functions import col, lit
from pyspark.sql.types import TimestampType

from src.config import logger


def _create_db(spark: SparkSession, database_name: str, location: str) -> str:
    try:
        if not _db_exists(spark, database_name):
            logger.info(f"Creating db {database_name} at {location}")
            database_path = f"CREATE DATABASE `{database_name}` LOCATION '{location}{database_name}/_db/'"  # noqa
            spark.sql(database_path)
            logger.info(
                f"Successfully created db {database_name} at {location}"
            )
        else:
            logger.info(f"Database {database_name} exists")
    except Exception as error:
        logger.exception(f"Failed to create db at {location}{database_name}")
        raise error


def _create_schema(
    db_workspace: WorkspaceClient, catalog_name: str, schema_name: str
):
    schema_full_name = f"{catalog_name}.{schema_name}"
    try:
        db_workspace.schema.get(schema_full_name)
        logger.info(f"Schema ({schema_full_name}) already exists")
        return None
    except Exception:
        logger.info(f"Creating Schema: {schema_full_name}")
        return db_workspace.schema.create(
            name=schema_name, catalog_name=catalog_name
        )


def create_abfss_path(
    container: str, storage_account: str = None, path: str = None
) -> str:
    # TODO: use urllib to join paths
    storage_account = (
        storage_account if storage_account else os.getenv("STORAGE_ACCOUNT")
    )
    abfss_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/"
    if path:
        if path[:1] == "/":
            path = path[1:]
        if path[-1:] == "/":
            path = path[:-1]
        abfss_path = f"{abfss_path}{path}/"
    return abfss_path


def save_new_table(
    df: DataFrame,
    table_full_name: str,
    file_format: str,
    mode="append",
    merge_schema=False,
    partition_by: Union[str, List[str], None] = None,
) -> None:
    logger.info(f"Save table to {table_full_name}")
    df.write.format(file_format).saveAsTable(
        table_full_name,
        mode=mode,
        partitionBy=partition_by,
        mergeSchema=merge_schema,
    )
    logger.info(f"Table successfully saved: {table_full_name}")


def _db_exists(spark: SparkSession, database: str) -> None:
    return spark.catalog._jcatalog.databaseExists(database)


def table_exists(db_workspace: WorkspaceClient, table_full_name: str):
    try:
        return db_workspace.table.get(table_full_name)
    except Exception:
        return None


def create_when_matched_update_condition(
    dataframe, removed_columns: list, conditions=None
):
    col_names = dataframe.columns
    for col_name in removed_columns:
        col_names.remove(col_name)

    for col_name in col_names:
        if conditions is None:
            conditions = col(f"src.{col_name}") != col(f"tgt.{col_name}")
        else:
            conditions = conditions | (
                col(f"src.{col_name}") != col(f"tgt.{col_name}")
            )

    return conditions


def create_update_dict(columns_list, excluded_columns: list = None):
    excluded_columns = excluded_columns if excluded_columns else []
    columns_dict = {}
    for column in columns_list:
        if column not in excluded_columns:
            tgt_column = f"tgt.{column}"
            src_column = f"src.{column}"
            columns_dict[tgt_column] = src_column
    return columns_dict


def create_merge_condition(match_cols: List):
    condition = ""
    for index in range(0, len(match_cols)):
        if index == 0:
            condition = f"src.{match_cols[index]} = tgt.{match_cols[index]}"
        else:
            condition = f"{condition} AND src.{match_cols[index]} = tgt.{match_cols[index]}"
    return condition


def save(
    spark: SparkSession,
    db_workspace: WorkspaceClient,
    df: DataFrame,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    file_format: str,
    mode: str,
    optimize_table: bool = False,
    partition_by: Union[List[str], str, None] = None,
    upsert_config: dict = None,
) -> None:
    df = df.withColumn(
        "processing_ts", lit(datetime.now()).cast(TimestampType())
    )

    table_full_name = f"{catalog_name}.{schema_name}.{table_name}"
    _create_schema(db_workspace, catalog_name, schema_name)
    logger.info(f"Preparing to write {table_full_name}")

    if mode == "upsert" and table_exists(db_workspace, table_full_name):
        when_matched_update_cond = create_when_matched_update_condition(
            df, upsert_config["exclusion_list"]
        )
        update_matched_col_set = create_update_dict(
            df.columns, upsert_config["exclusion_list"]
        )
        update_not_matched_col_set = create_update_dict(df.columns)
        logger.info(f"Upserting : {table_full_name}")
        delta_table = DeltaTable.forName(spark, table_full_name)
        delta_table.alias("tgt").merge(
            df.alias("src"),
            condition=create_merge_condition(upsert_config["on_match_cols"]),
        ).whenMatchedUpdate(
            condition=when_matched_update_cond,
            set=update_matched_col_set,
        ).whenNotMatchedInsert(
            values=update_not_matched_col_set
        ).execute()
    else:
        if mode == "upsert":
            mode = "append"
        save_new_table(
            df,
            table_full_name,
            file_format,
            mode=mode,
            partition_by=partition_by,
        )
    if optimize_table:
        logger.info("Applying table optimizer")
        optimize_query = f"OPTIMIZE {table_full_name}"
        spark.sql(optimize_query)


def create_database_name(src_database, dst_database, git_hash=None):
    if git_hash is not None:
        src_database = f"{src_database}_{git_hash}"
        dst_database = f"{dst_database}_{git_hash}"
    return src_database, dst_database
