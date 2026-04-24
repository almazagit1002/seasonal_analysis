import yaml
import os
from utils.logger import get_logger

logger = get_logger("config_reader", log_to_file=False)

class ConfigReader:
    def __init__(self, config_file):
        """
        Reads a YAML configuration file.
        config_file: str, name of the YAML file without extension
        """
        # Adjusted path to project root
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", f"{config_file}.yaml")
        path = os.path.abspath(path)  # get absolute path

        if not os.path.exists(path):
            logger.error(f"Config file not found: {path}")
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")

    def get(self, key, default=None):
        """Get any top-level key from the config"""
        return self.config.get(key, default)