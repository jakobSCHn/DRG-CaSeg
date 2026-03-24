import yaml
import attrs
import copy
import numpy as np

from datetime import datetime
from pathlib import Path
from utils import save_dict_to_yaml, check_filepaths
from infra.infra_utils import DataStage, configure_callable, setup_experiment_folder

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
        #Check whether required keys for the pipeline are present
        if "experiment_name" not in cfg or "dataset" not in cfg:
            raise KeyError(
                f"Config file must define 'experiment_name' and " 
                f"'dataset' to run the pipeline."
            )
        
        return cls(name=cfg["experiment_name"], config=cfg, config_path=path)


    def _load_data(
        self,
        data_cfg: dict,
        runtime_context: dict,
        ):
        loader = configure_callable(
            id=data_cfg["id"],
            import_path=data_cfg["loader"],
            params=data_cfg.get("params", {}),
            context=runtime_context, 
        )

        payload = loader()

        data = payload["data"]
        md = data.meta_data[0]
        gt = payload.get("gt", {})
        gt["fps"] = data.fr

        background_img = payload["background_img"]

        return data, md, gt, background_img
    

    def _preprocess(
        self,
        data,
        steps: list,
        runtime_context: dict,
        ):

        for step_pre in steps:
            logger.info(f"Preprocessing: {step_pre["id"]}")
            preprocessor = configure_callable(
                id=step_pre["id"],
                import_path=step_pre["function"],
                params=step_pre.get("params", {}),
                context=runtime_context,
            )
            data = preprocessor(data)

        return data


    def _analyze(
        self,
        analysis_method: dict,
        data,
        save_path: Path,
        runtime_context: dict,
        ):

        analyzer = configure_callable(
            id=analysis_method["id"],
            import_path=analysis_method["function"],
            params={
                **analysis_method.get("params", {}),
                "save_filepath": save_path,
            },
            context=runtime_context,
        )

        results = analyzer(data)

        if analysis_method.get("export_matrices", False):
            logger.info("Exporting data in numpy matrix format")
            np.savez_compressed(
                save_path / "raw_matrices.npz",
                masks=results["masks"],
                traces=results["traces"],
            )
        
        return results
    

    def _postprocess(
        self,
        results: dict,
        analysis_method: dict,
        steps: list,
        background_img: np.ndarray,
        fr: float,
        save_path: Path,
        runtime_context: dict,  
        ):

        results_post = copy.deepcopy(results)
        for step_post in steps:
            logger.info(f"Applying postprocessing with ID: {step_post["id"]}")
            postprocessor = configure_callable(
                id=step_post["id"],
                import_path=step_post["function"],
                params={
                    **step_post.get("params", {}),
                    "background_img": background_img,
                    "fr": fr
                },
                context=runtime_context,
            )
            results_post = postprocessor(results_post)
            
        if analysis_method.get("export_matrices", False):
            np.savez_compressed(
                save_path / "postprocessed_matrices.npz",
                masks=results_post["masks"],
                traces=results_post["traces"],
                sampling_frequency=fr,
            )

        return results_post
    

    def _visualize(
        self,
        visualizations,
        results,
        data,
        gt,
        background_img,
        stage,
        save_path,
        runtime_context,
        ):
        for vis in visualizations:
            logger.info(f"Plotting results with plotting ID: {vis["id"]}")
            plotter = configure_callable(
                id=vis["id"],
                import_path=vis["function"],
                params={
                    **vis.get("params", {}),
                    "results": results,
                    "data": data,
                    "gt": gt,
                    "background_img": background_img,
                    "stage": stage,
                    "save_filepath": save_path,
                },
                context=runtime_context,
            )
            plotter()


    def _evaluate(
        self,
        evaluation_methods,
        metrics,
        results,
        gt,
        md,
        save_path,
        runtime_context,
        ):
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

            metrics.update(evaluator(res=results, gt=gt, md=md))
            save_dict_to_yaml(metrics, save_path=save_path / f"metrics_{eva["id"]}.yaml")
    

    def run(
            self,
            runtime_context: dict = {}
        ):
        logger.info(f"Running: {self.config["experiment_name"]}")
        
        #Load experiment configurations
        data_cfgs = self.config["dataset"]
        preprocessing = self.config.get("preprocessing", [])
        analysis_methods = self.config["analysis"]
        postprocessing = self.config.get("postprocessing", [])
        visualizations = self.config.get("visualization", [])
        evaluation_methods = self.config.get("evaluation", [])

        #Safety check to see whether all filenames exist that have been referenced
        #before iterating through them
        check_filepaths(data_cfgs)

        #Load the data samples for the experiment and analyze them one
        #after another to reduce RAM needs
        for data_cfg in data_cfgs:

            logger.info(f"Loading dataset ID: {data_cfg["id"]}")

            data, md, gt, background_img = self._load_data(
                data_cfg=data_cfg,
                runtime_context=runtime_context,
            )

            #Preprocess the loaded data
            if preprocessing:
                data = self._preprocess(
                    data=data,
                    steps=preprocessing,
                    runtime_context=runtime_context,
                )
            else:
                logger.warning(f"No preprocessing configured.")

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

                results = self._analyze(
                    analysis_method=ana,
                    data=data,
                    save_path=save_path,
                    runtime_context=runtime_context
                )
                metrics = results.get("analysis_stats", {})
                
                if postprocessing:
                    results_post = self._postprocess(
                        results=results,
                        analysis_method=ana,
                        steps=postprocessing,
                        background_img=background_img,
                        fr=data.fr,
                        save_path=save_path,
                        runtime_context=runtime_context
                    )
                else:
                    results_post = {}
                    logger.warning(f"No postprocessing configured.")


                if visualizations:
                    for data_stage, res in [(DataStage.RAW, results), (DataStage.POST_PROCESSED, results_post)]:
                        if not res:
                            continue
                        self._visualize(
                            visualizations=visualizations,
                            results=res,
                            data=data,
                            gt=gt,
                            background_img=background_img,
                            stage=data_stage,
                            save_path=save_path,
                            runtime_context=runtime_context,
                        )
                else:
                    logger.warning(f"No visualization configured.")


                if evaluation_methods and {"spatial", "temporal"} <= gt.keys():
                    self._evaluate(
                        evaluation_methods=evaluation_methods,
                        metrics=metrics,
                        results=results,
                        gt=gt,
                        md=md,
                        save_path=save_path,
                        runtime_context=runtime_context,
                    )
                else:
                    logger.warning("No Evaluation configured.")

        logger.info(f"Experiment {self.name} has been completed successfully!")