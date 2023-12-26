from databricks.sdk.service.compute import (
    Library,
    PythonPyPiLibrary,
    LibrariesAPI,
)
from typing import List, Dict, Tuple
from src.utils import load_yaml, run_in_databricks
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.connect import DatabricksSession
import argparse


def extract_packages(requirements_paths: List[str]):
    requirements: List[Dict[str, str]] = []
    for requirements_file in requirements_paths:
        with open(requirements_file, "r") as text_stream:
            requirements.extend(
                [
                    Library(pypi=PythonPyPiLibrary(library.replace("\n", "")))
                    for library in text_stream.readlines()
                ]
            )
    return requirements


def update_cluster_packages_from_config(
    requirements_paths: List[str], databricks_connect: Dict[str, str], **kwargs
):
    requirements = extract_packages(requirements_paths)
    _, db_client = get_spark_session_db_client()
    library_api = LibrariesAPI(db_client.api_client)
    cluster_id = databricks_connect["cluster_id"]
    for cluster_id in databricks_connect["cluster_id"]:
        library_api.install(cluster_id, requirements)
    return True


def get_spark_session_db_client() -> Tuple[SparkSession, WorkspaceClient]:
    if run_in_databricks():
        builder = SparkSession.builder
    else:
        databricks_connect_config = load_yaml(
            "./deploy/env_configs/local.yaml"
        )
        databricks_profile_name = databricks_connect_config[
            "databricks_connect"
        ]["profile"]
        builder = DatabricksSession.builder.profile(databricks_profile_name)
    db_client = WorkspaceClient()
    return builder.getOrCreate(), db_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("-c", "--config-file", required=True)

    args = parser.parse_args()
    yaml_path = args.config_file

    config = load_yaml(yaml_path)
    update_cluster_packages_from_config(**config)
