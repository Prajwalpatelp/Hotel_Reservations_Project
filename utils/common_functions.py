import os
import pandas
from Src.logger import get_logger
from Src.custom_exception import CustomException
import yaml
import pandas as pd


logger = get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Succesfully Read the yaml file")
            return config
        
    except Exception as e:
        logger.error("Error while reading the yaml file")
        raise CustomException(f"Failed to read YAML file",e)
    

# loading csv file
def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.info(f"Error During loading the data {e}")
        raise CustomException("Failed to load data",e)