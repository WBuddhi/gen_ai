import os

from src.config import db_client, logger, spark
from src.utils import load_yaml, run_in_databricks
from mlflow import create_experiment, log_params, start_run, get_experiment_by_name, autolog
from mlflow.exceptions import MlflowException
from datasets import IterableDataset


class PLSTrainer:
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.__dict__.update(self.config["feature_store"])
        self.__dict__.update(self.config["modelling"])
        self.experiment_id = self.mlflow_setup()
        autolog()

    def mlflow_setup(self):
        try:
            return create_experiment(self.experiment_name)
        except MlflowException:
            logger.info("Experiment already exists")
            return get_experiment_by_name(self.experiment_name).experiment_id

    def load_dataset(self):
        logger.info("Loading data from feature store")
        full_schema_name = f"{self.catalog_name}.{self.schema_name}"
        datasets = {}
        for table_name in ("train", "validation", "test"):
            datasets[table_name] = IterableDataset.from_spark(
                spark.table(f"{full_schema_name}.{table_name}")
            )
        return datasets

    def train(self):
        pass

    def run(self):
        with start_run(experiment_id=self.experiment_id):
            log_params(self.config)
            self.load_dataset()
            self.train()


if __name__ == "__main__":
    config_relative_path = "src/pipeline_configs/llama2_7b_pls.yaml"
    config_path = (
        os.path.join(os.environ["REPO_ROOT_PATH"], config_relative_path)
        if run_in_databricks()
        else os.path.join(".", config_relative_path)
    )
    pls_trainer = PLSTrainer(config_path=config_path)
    pls_trainer.run()
