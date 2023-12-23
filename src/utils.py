import yaml
import os


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as yaml_stream:
        return yaml.safe_load(yaml_stream)


def run_in_databricks():
    if os.environ.get("REPO_ROOT_PATH", None):
        return True
    return False
