import yaml
import os


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as yaml_stream:
        return yaml.safe_load(yaml_stream)


def run_in_databricks():
    if os.environ.get("REPO_ROOT_PATH", None):
        return True
    return False


def get_path_to_src():
    cwd = os.getcwd().split("/")
    path_to_src = []
    for folder_name in cwd[::-1]:
        if folder_name == "gen_ai":
            break
        path_to_src.append("..")
    return "/".join(path_to_src)
