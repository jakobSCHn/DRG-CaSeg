import yaml
import attrs
import os


from infra.utils import configure_callable
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

@attrs.define
class Experiment:
    config: dict
    run_id: str = attrs.field(factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        #Saftey check on whether a valid file was passed
        if cfg is None:
            raise ValueError(f"The config file at {path} is empty.")
        
        return cls(config=cfg)


    def run(self):
        logger.info(f"Running: {self.config['experiment_name']}")
        data_cfgs = self.config["dataset"]
        steps_pre = self.config.get("preprocessing", [])

        for data_cfg in data_cfgs:
            #Load the data for the experiment
            logger.info(f"Loading dataset ID: {data_cfg["id"]}")
            loader = configure_callable(data_cfg["loader"], data_cfg.get("params", {}))
            payload = loader()

            data = payload["data"]

            #Preprocess the loaded data
            if steps_pre:
                for step in steps_pre:
                    logger.info(f"Preprocessing: {step["name"]}")
                    processor = configure_callable(step["function"], step.get("params", {}))
                    data = processor(data)
            else:
                logger.warning(f"No Data preprocessing applied.")

            #Data Analysis
            
        
        #Saving Results
        self._save_results(model_instance, current_data)