import yaml
import attrs

from datetime import datetime
from infra.utils import configure_callable, setup_experiment_folder, save_dict_to_yaml
from data_utils.plotter import render_inference_video

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
        
        #Load experiment configurations
        data_cfgs = self.config["dataset"]
        steps_pre = self.config.get("preprocessing", [])
        analysis_methods = self.config["analysis"]
        evaluation_methods = self.config.get("evaluation", [])

        for data_cfg in data_cfgs:
            #Load the data for the experiment
            logger.info(f"Loading dataset ID: {data_cfg["id"]}")
            loader = configure_callable(data_cfg["loader"], data_cfg.get("params", {}))
            payload = loader()

            data = payload["data"]
            gt = payload["gt"]

            #Preprocess the loaded data
            if steps_pre:
                for step in steps_pre:
                    logger.info(f"Preprocessing: {step["name"]}")
                    processor = configure_callable(step["function"], step.get("params", {}))
                    data = processor(data)
            else:
                logger.warning(f"No Data preprocessing applied.")

            #Data Analysis
            for ana in analysis_methods:    
                logger.info(f"Analyzing data with Analysis ID: {ana["id"]}")
                analyzer = configure_callable(ana["function"], ana.get("params", {}))
                results = analyzer(data)

                save_path = setup_experiment_folder(
                    run_id=self.run_id,
                    config_path=self.config,
                    ana_id=ana["id"],
                )

                render_inference_video(
                    roi_masks=results["masks"],
                    roi_traces=results["traces"],
                    video_data=data,
                    save_filepath=save_path / "inference.mp4"
                )

                if evaluation_methods:
                    for eva in evaluation_methods:
                        logger.info(f"Evaluating with Evaluation ID: {eva["id"]}")
                        evaluator = configure_callable(eva["function"], eva.get("params", {}))
                        metrics = evaluator(pred=results["masks"], gt=gt)
                        logger.info(f"Evaluation Results for {data_cfg['id']}:")
                        logger.info(f"  Precision: {metrics['precision']:.4f}")
                        logger.info(f"  Mean IoU:  {metrics['mean_iou']:.4f}")

                        save_dict_to_yaml(metrics, save_path=save_path / "metrics.yaml")
                else:
                    logger.warning("No Evaluation configured.")

        logger.info(f"Experiment {self.run_id} has been completed successfully!")