import yaml
import os


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as yaml_stream:
        return yaml.safe_load(yaml_stream)


def run_in_databricks():
    if os.environ.get("DB_CLUSTER_NAME", None):
        return True
    return False


def get_path_to_src():
    cwd = os.getcwd().split("/")
    path_to_src = []
    if "src" not in cwd:
        return "."
    for folder_name in cwd[::-1]:
        path_to_src.append("..")
        if folder_name == "src":
            break
    return "/".join(path_to_src)
