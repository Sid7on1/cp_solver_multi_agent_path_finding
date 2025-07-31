# data_loader.py
"""
Module for loading and preprocessing dynamic agent trajectories.
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing dynamic agent trajectories.
    """

    def __init__(self, config: Config):
        """
        Initialize the DataLoader with the given configuration.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.scaler = StandardScaler()

    def load_trajectories(self, data_dir: str) -> pd.DataFrame:
        """
        Load the dynamic agent trajectories from the given directory.

        Args:
            data_dir (str): The directory containing the trajectory data.

        Returns:
            pd.DataFrame: The loaded trajectory data.
        """
        try:
            # Load the trajectory data from the CSV files
            trajectory_data = []
            for file in os.listdir(data_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(data_dir, file)
                    trajectory = pd.read_csv(file_path)
                    trajectory_data.append(trajectory)
            # Concatenate the trajectory data into a single DataFrame
            trajectory_data = pd.concat(trajectory_data, ignore_index=True)
            return trajectory_data
        except FileNotFoundError as e:
            logger.error(f"Error loading trajectory data: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Error loading trajectory data: {e}")
            raise

    def preprocess_data(self, trajectory_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the trajectory data by scaling and splitting it into training and testing sets.

        Args:
            trajectory_data (pd.DataFrame): The trajectory data to preprocess.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The preprocessed training and testing data.
        """
        try:
            # Scale the trajectory data using StandardScaler
            scaled_data = self.scaler.fit_transform(trajectory_data)
            scaled_data = pd.DataFrame(scaled_data, columns=trajectory_data.columns)

            # Split the scaled data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(scaled_data.drop("target", axis=1), scaled_data["target"], test_size=self.config.test_size, random_state=self.config.random_state)

            return X_train, X_test, y_train, y_test
        except ValueError as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

class Config:
    """
    Class for storing configuration settings.
    """

    def __init__(self):
        """
        Initialize the Config object with default settings.
        """
        self.test_size = 0.2
        self.random_state = 42

def main():
    # Load the configuration
    config = Config()

    # Create a DataLoader instance
    data_loader = DataLoader(config)

    # Load the trajectory data
    data_dir = "path/to/data/directory"
    trajectory_data = data_loader.load_trajectories(data_dir)

    # Preprocess the trajectory data
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(trajectory_data)

    # Print the preprocessed data
    print("Preprocessed Training Data:")
    print(X_train.head())
    print("Preprocessed Testing Data:")
    print(X_test.head())

if __name__ == "__main__":
    main()