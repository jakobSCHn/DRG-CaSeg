import json
import yaml
import attrs
import functools
import os


from infra.utils import get_object_from_path
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
            data = yaml.safe_load(f)
        return cls(config=data)


    def _execute_component(
        self,
        path,
        params,
        input_data=None
        ):
        """
        Helper to load code and execute it.
        Handles the difference between a Class (needs init) and a Function.
        """
        obj_def = get_object_from_path(path)
        
        if input_data is not None:
            return obj_def(input_data, **params)
        else:
            return obj_def(**params)
            

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


    def run(self):
        logger.info(f"Running: {self.config['experiment_name']}")
        
        # 1. Data Loading
        data_conf = self.config["dataset"]
        logger.info(f"Loading data: {data_conf['path']}")
        current_data = self._execute_component(data_conf["path"], data_conf.get("params", {}))
        
        # 2. Preprocessing Pipeline
        pipeline_steps = self.config.get("preprocessing", [])
        for step in pipeline_steps:
            print(f"Preprocessing: {step['path']}")
            current_data = self._execute_component(
                step["path"], 
                step.get("params", {}), 
                input_data=current_data
            )

        # 3. Model Training
        model_conf = self.config["model"]
        print(f"Training model: {model_conf['path']}")
        
        # Here we assume the model returns the object itself after fitting
        # or we might want to capture the labels depending on your logic
        model_obj = get_object_from_path(model_conf["path"])
        model_instance = model_obj(**model_conf.get("params", {}))
        
        # Fit the model
        model_instance.fit(current_data)
        
        # 4. Save Results
        self._save_results(model_instance, current_data)

    def _save_results(self, model, final_data):
        base_dir = self.config.get("output_dir", "./experiments")
        save_path = os.path.join(base_dir, self.config["experiment_name"], self.run_id)
        os.makedirs(save_path, exist_ok=True)
        
        
        # Save Config
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        print(f"Saved to: {save_path}")