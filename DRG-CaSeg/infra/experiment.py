import yaml
import attrs
import numpy as np

from datetime import datetime
from utils import save_dict_to_yaml
from infra.infra_utils import configure_callable, setup_experiment_folder

import logging
logger = logging.getLogger(__name__)

@attrs.define
class Experiment:
    name: str
    config: dict
    config_path: str
    run_id: str = attrs.field(factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        #Saftey check on whether a valid file was passed
        if cfg is None:
            raise ValueError(f"The config file at {path} is empty.")
        
        return cls(name=cfg["experiment_name"], config=cfg, config_path=path)


    def run(
            self,
            runtime_context: dict = None,
        ):
        logger.info(f"Running: {self.config["experiment_name"]}")
        
        #Load experiment configurations
        data_cfgs = self.config["dataset"]
        steps_pre = self.config.get("preprocessing", [])
        analysis_methods = self.config["analysis"]
        visualization = self.config.get("visualization", [])
        evaluation_methods = self.config.get("evaluation", [])

        for data_cfg in data_cfgs:
            #Load the data for the experiment
            logger.info(f"Loading dataset ID: {data_cfg["id"]}")
            loader = configure_callable(
                id=data_cfg["id"],
                import_path=data_cfg["loader"],
                params=data_cfg.get("params", {}),
                context=runtime_context,
            )
            payload = loader()

            data = payload["data"]
            md = payload.get("meta", [])
            gt = payload.get("gt", [])
            gt["fps"] = data.fr

            background_img = np.percentile(data, 98, axis=0)

            #Preprocess the loaded data
            if steps_pre:
                for step in steps_pre:
                    logger.info(f"Preprocessing: {step["id"]}")
                    processor = configure_callable(
                        id=step["id"],
                        import_path=step["function"],
                        params=step.get("params", {}),
                        context=runtime_context,
                    )
                    data = processor(data)
            else:
                logger.warning(f"No Data preprocessing applied.")

            #Data Analysis
            for ana in analysis_methods:    
                logger.info(f"Analyzing data with Analysis ID: {ana["id"]}")

                save_path = setup_experiment_folder(
                    experiment_name=self.name,
                    run_id=self.run_id,
                    config_path=self.config_path,
                    data_id=data_cfg["id"],
                    ana_id=ana["id"],
                )

                analyzer = configure_callable(
                    id=ana["id"],
                    import_path=ana["function"],
                    params={
                        **ana.get("params", {}),
                        "save_filepath": save_path,
                    },
                    context=runtime_context,
                )
                results = analyzer(data)


                if visualization:
                    for vis in visualization:
                        logger.info(f"Plotting results with Results ID: {vis["id"]}")
                        plotter = configure_callable(
                            id=vis["id"],
                            import_path=vis["function"],
                            params={
                                **vis.get("params", {}),
                                "results": results,
                                "data": data,
                                "gt": gt,
                                "background_img": background_img,
                                "save_filepath": save_path,
                            },
                            context=runtime_context,
                        )
                        plotter()

                metrics = results.get("analysis_stats", {})

                if evaluation_methods and len(gt) > 0:
                    for eva in evaluation_methods:
                        logger.info(f"Evaluating with Evaluation ID: {eva["id"]}")
                        evaluator = configure_callable(
                            id=eva["id"],
                            import_path=eva["function"],
                            params={
                                **eva.get("params", {}),
                                "save_filepath": save_path,
                            },
                            context=runtime_context,
                        )

                        metrics.update(evaluator(res=results, gt=gt))
                        save_dict_to_yaml(metrics, save_path=save_path / "metrics.yaml")

                else:
                    logger.warning("No Evaluation configured.")

        logger.info(f"Experiment {self.name} has been completed successfully!")