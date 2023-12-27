from datetime import datetime
from typing import Dict, List, Union

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import TimestampType

from src.config import logger


def _create_schema(
    db_client: WorkspaceClient, catalog_name: str, schema_name: str
):
    schema_full_name = f"{catalog_name}.{schema_name}"
    try:
        db_client.schemas.get(schema_full_name)
        logger.info(f"Schema ({schema_full_name}) already exists")
        return None
    except Exception:
        logger.info(f"Creating Schema: {schema_full_name}")
        return db_client.schemas.create(
            name=schema_name, catalog_name=catalog_name
        )


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


def table_exists(db_client: WorkspaceClient, table_full_name: str):
    try:
        return db_client.table.get(table_full_name)
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


def upsert_to_table(
    table_full_name: str, df: DataFrame, upsert_config: Dict[str, str]
):
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


def create_volume(
    db_client, catalog_name, schema_name, table_name, volume_type="MANAGED"
):
    _create_schema(db_client, catalog_name, schema_name)
    try:
        db_client.volumes.read(f"{catalog_name}.{schema_name}.{table_name}")
        logger.info("Volume already exists")
    except Exception:
        return db_client.volumes.create(
            catalog_name, schema_name, table_name, VolumeType(volume_type)
        )


def save_to_feature_store(
    table_name: str,
    df: DataFrame,
    mode: str,
    feature_store_client: FeatureEngineeringClient,
    primary_keys: List[str],
    partition_by: List[str] = None,
):
    try:
        feature_store_client.get_table(table_name)
        logger.info(f"{table_name}: Already exists")
        feature_store_client.write(
            name=table_name,
            df=df,
            mode=mode,
        )
    except Exception:
        logger.info(f"{table_name}: doesn't exist")
        feature_store_client.create_table(
            name=table_name,
            df=df,
            primary_keys="id",
            partition_columns=partition_by,
        )


def save(
    spark: SparkSession,
    db_client: WorkspaceClient,
    df: DataFrame,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    file_format: str,
    mode: str,
    primary_keys: List[str] = None,
    optimize_table: bool = False,
    partition_by: Union[List[str], str, None] = None,
    upsert_config: dict = None,
    feature_store_client: FeatureEngineeringClient = None,
) -> None:
    df = df.withColumn(
        "processing_ts", lit(datetime.now()).cast(TimestampType())
    )

    table_full_name = f"{catalog_name}.{schema_name}.{table_name}"
    _create_schema(db_client, catalog_name, schema_name)
    logger.info(f"Preparing to write {table_full_name}")
    if file_format == "VOLUME":
        create_volume(db_client, catalog_name, schema_name, table_name)
        pass
    elif file_format == "FEATURESTORE":
        save_to_feature_store(
            table_name=table_full_name,
            df=df,
            primary_keys=primary_keys,
            partition_by=partition_by,
            mode=mode,
            feature_store_client=feature_store_client,
        )
    elif mode == "upsert" and table_exists(db_client, table_full_name):
        upsert_to_table(table_full_name, df, upsert_config)
    else:
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
