import yaml


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as yaml_stream:
        return yaml.safe_load(yaml_stream)
