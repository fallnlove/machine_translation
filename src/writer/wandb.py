import numpy as np
import pandas as pd

from src.metrics.tracker import MetricTracker


class WanDBWriter:
    def __init__(
        self,
        project_name: str,
        run_name: str = None,
        run_id: str = None,
        start_step: int = 0,
        **kwargs,
    ):
        """
        Initialize the WanDBWriter

        Input:
            project_name (str): Name of the project
            run_name (str): Name of the run
            run_id (str): ID of the run
            start_step (int): number of step to start
        """
        try:
            import wandb

            wandb.init(
                project=project_name, name=run_name, id=run_id, resume="allow", **kwargs
            )

            self.wandb = wandb

        except ImportError:
            raise ImportError("Please install wandb to use this writer")

        self.step_ = start_step
        self.mode = ""

    def step(self):
        """
        Step the writer
        """

        self.step_ += 1

    def train(self):
        """
        Set the writer to train mode
        """

        self.mode = "train"

    def eval(self):
        """
        Set the writer to eval mode
        """

        self.mode = "eval"

    def log_scalar(self, name: str, value: float):
        """
        Log a scalar value

        Input:
            name (str): Name of the scalar
            value (float): Value of the scalar
        """

        self.wandb.log({f"{name}_{self.mode}": value}, step=self.step_)

    def log_image(self, name: str, image: np.ndarray):
        """
        Log an image

        Input:
            name (str): Name of the image
            image (np.ndarray): Image to log
        """

        self.wandb.log(
            {f"{name}_{self.mode}": self.wandb.Image(image)}, step=self.step_
        )

    def log_metrics(self, metrics: MetricTracker):
        """
        Log metrics

        Input:
            metrics (MetricsTracker): Metrics to log
        """

        for key in metrics.keys():
            self.log_scalar(key, metrics.get(key))
    
    def log_table(self, name: str, table: pd.DataFrame):
        """
        Log the table

        Input:
            name (str): Name of the table
            table (pd.DataFrame): Table to log
        """

        self.wandb.log(
            {f"{name}_{self.mode}": self.wandb.Table(dataframe=table)},
            step=self.step_,
        )
