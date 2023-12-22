from databricks.sdk.service.compute import (
    Library,
    PythonPyPiLibrary,
    LibrariesAPI,
)
from typing import List, Dict, Union
from src.utils import load_yaml
from src.ingestion.spark_config import get_spark_session_db_client
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
    _, db_client = get_spark_session_db_client(
        name="deploy", databricks_connect=databricks_connect
    )
    cluster_id = databricks_connect["cluster_id"]
    library_api = LibrariesAPI(db_client.api_client)
    return library_api.install(cluster_id, requirements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("-c", "--config-file", required=True)

    args = parser.parse_args()
    yaml_path = args.config_file

    config = load_yaml(yaml_path)
    update_cluster_packages_from_config(**config)
