# utils.py
"""
Utility functions for the CP-Solver application.
"""

import logging
import numpy as np
import torch
from typing import Tuple, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Configuration:
    """
    Configuration class for the CP-Solver application.
    """

    def __init__(self, settings: Dict):
        """
        Initialize the configuration.

        Args:
            settings (Dict): Configuration settings.
        """
        self.settings = settings

    @property
    def distance_metric(self) -> str:
        """
        Get the distance metric.

        Returns:
            str: Distance metric.
        """
        return self.settings.get("distance_metric", "euclidean")

    @property
    def collision_threshold(self) -> float:
        """
        Get the collision threshold.

        Returns:
            float: Collision threshold.
        """
        return self.settings.get("collision_threshold", 0.5)


class DistanceCalculator:
    """
    Distance calculator class.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the distance calculator.

        Args:
            config (Configuration): Configuration.
        """
        self.config = config

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the distance between two points.

        Args:
            point1 (Tuple[float, float]): First point.
            point2 (Tuple[float, float]): Second point.

        Returns:
            float: Distance.
        """
        if self.config.distance_metric == "euclidean":
            return np.linalg.norm(np.array(point1) - np.array(point2))
        elif self.config.distance_metric == "manhattan":
            return np.sum(np.abs(np.array(point1) - np.array(point2)))
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")


class CollisionDetector:
    """
    Collision detector class.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the collision detector.

        Args:
            config (Configuration): Configuration.
        """
        self.config = config

    def check_collision(self, agent1: Dict, agent2: Dict) -> bool:
        """
        Check if two agents collide.

        Args:
            agent1 (Dict): First agent.
            agent2 (Dict): Second agent.

        Returns:
            bool: Collision status.
        """
        distance = self.calculate_distance(agent1["position"], agent2["position"])
        return distance < self.config.collision_threshold

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the distance between two points.

        Args:
            point1 (Tuple[float, float]): First point.
            point2 (Tuple[float, float]): Second point.

        Returns:
            float: Distance.
        """
        return DistanceCalculator(self.config).calculate_distance(point1, point2)


class Utils:
    """
    Utility class for the CP-Solver application.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the utility class.

        Args:
            config (Configuration): Configuration.
        """
        self.config = config
        self.distance_calculator = DistanceCalculator(config)
        self.collision_detector = CollisionDetector(config)

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the distance between two points.

        Args:
            point1 (Tuple[float, float]): First point.
            point2 (Tuple[float, float]): Second point.

        Returns:
            float: Distance.
        """
        return self.distance_calculator.calculate_distance(point1, point2)

    def check_collision(self, agent1: Dict, agent2: Dict) -> bool:
        """
        Check if two agents collide.

        Args:
            agent1 (Dict): First agent.
            agent2 (Dict): Second agent.

        Returns:
            bool: Collision status.
        """
        return self.collision_detector.check_collision(agent1, agent2)


def main():
    """
    Main function.
    """
    config = Configuration({
        "distance_metric": "euclidean",
        "collision_threshold": 0.5
    })
    utils = Utils(config)

    agent1 = {
        "position": (0.0, 0.0),
        "velocity": (1.0, 1.0)
    }
    agent2 = {
        "position": (1.0, 1.0),
        "velocity": (2.0, 2.0)
    }

    distance = utils.calculate_distance(agent1["position"], agent2["position"])
    logger.info(f"Distance: {distance}")

    collision = utils.check_collision(agent1, agent2)
    logger.info(f"Collision: {collision}")


if __name__ == "__main__":
    main()