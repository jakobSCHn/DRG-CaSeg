import yaml
import attrs
import os
import inspect


from infra.utils import get_object_from_path, configure_callable
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

            

    def _save_results(
        self,
        model,
        final_data
        ):
        base_dir = self.config.get("output_dir", "./experiments")
        save_path = os.path.join(base_dir, self.config["experiment_name"], self.run_id)
        os.makedirs(save_path, exist_ok=True)
        
        
        # Save Config
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        print(f"Saved to: {save_path}")

    
    def _save_results(self, model, final_data):
        base_dir = self.config.get("output_dir", "./experiments")
        save_path = os.path.join(base_dir, self.config["experiment_name"], self.run_id)
        os.makedirs(save_path, exist_ok=True)
        
        
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        logger.info(f"Saved to: {save_path}")


    def _save_results(self, model, final_data):
        base_dir = self.config.get("output_dir", "./experiments")
        save_path = os.path.join(base_dir, self.config["experiment_name"], self.run_id)
        os.makedirs(save_path, exist_ok=True)
        
        
        # Save Config
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        print(f"Saved to: {save_path}")


    def run(self):
        logger.info(f"Running: {self.config['experiment_name']}")
        
        #Load the data for the experiment
        data_conf = self.config["dataset"]
        logger.info(f"Loading data...")
        loader = configure_callable(data_conf["loader"], data_conf.get("params", {}))
        data = loader()

        #Preprocess the loaded data
        pipeline_steps = self.config.get("preprocessing", [])
        if pipeline_steps:
            for step in pipeline_steps:
                logger.info(f"Preprocessing: {step["name"]}")
                processor = configure_callable(step["function"], step.get("params", {}))
                data = processor(data)
        else:
            logger.warning(f"No Data preprocessing applied.")

        #Model Training
        
        #Saving Results
        self._save_results(model_instance, current_data)