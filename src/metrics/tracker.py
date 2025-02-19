from collections import defaultdict


class MetricTracker:
    def __init__(self, keys):
        self.data = defaultdict()
        for key in keys:
            self.data[key] = {
                "values": [],
                "count": 0,
                "mean": 0,
            }

    def reset(self):
        """
        Reset all values to zero
        """
        for key in self.data.keys():
            self.data[key]["values"] = []
            self.data[key]["count"] = 0
            self.data[key]["mean"] = 0

    def update(self, key: str, value: float, n: int = 1):
        """
        Update the value of the key
        Input:
            key: str
            value: float
            n: int
        """
        self.data[key]["values"].append(value * n)
        self.data[key]["count"] += n
        self.data[key]["mean"] = sum(self.data[key]["values"]) / self.data[key]["count"]

    def get(self, key):
        """
        Get the mean value of the key
        Input:
            key: str
        """
        return self.data[key]["mean"]

    def mean(self):
        """
        Get the mean value of all keys
        """
        return {key: self.data[key]["mean"] for key in self.data.keys()}

    def keys(self):
        """
        Get the keys
        """
        return self.data.keys()
