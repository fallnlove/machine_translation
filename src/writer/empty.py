import numpy as np

from src.metrics.tracker import MetricTracker


class EmptyWriter:
    def __init__(
        self, project_name: str = None, run_name: str = None, run_id: str = None
    ):
        """
        Initialize the WanDBWriter

        Input:
            project_name (str): Name of the project
            run_name (str): Name of the run
            run_id (str): ID of the run
        """
        pass

    def step(self):
        """
        Step the writer
        """
        pass

    def train(self):
        """
        Set the writer to train mode
        """
        pass

    def eval(self):
        """
        Set the writer to eval mode
        """
        pass

    def log_scalar(self, name: str, value: float):
        """
        Log a scalar value

        Input:
            name (str): Name of the scalar
            value (float): Value of the scalar
        """
        pass

    def log_image(self, name: str, image: np.ndarray):
        """
        Log an image

        Input:
            name (str): Name of the image
            image (np.ndarray): Image to log
        """
        pass

    def log_metrics(self, metrics: MetricTracker):
        """
        Log metrics

        Input:
            metrics (MetricsTracker): Metrics to log
        """
        pass

    def log_table(self, *args, **kwargs):
        pass
