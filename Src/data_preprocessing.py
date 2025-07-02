import os
import pandas as pd
import numpy as np
from Src.logger import get_logger
from Src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Starting our Data Processing Steps")

            logger.info("Dropping the columns")
            df.drop(['Unnamed: 0','Booking_ID'],axis=1,inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Applying Label Encoding")

            Encoder=LabelEncoder()
            mappings={}
            for column in cat_cols:
                df[column]=Encoder.fit_transform(df[column])
                mappings[column]={label: code for label, code in zip(Encoder.classes_,Encoder.transform(Encoder.classes_))}

            logger.info("Label Mappings are:")
            for col,mappings in mappings.items():
                logger.info(f"{col} : {mappings}")


            logger.info("Handling Skewness")

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x:x.skew())


            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        except Exception as e:
            logger.info(f"Error During processing Steps {e}")
            raise CustomException(f"Error While processing data",e)
    
    def balance_data(self,df):
        try:
            logger.info("Handling Im-balanced Data")
            X = df.drop(columns='booking_status',axis=1)
            y=df['booking_status']
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_resampled,columns=X.columns)
            balanced_df['booking_status'] = y_resampled


            logger.info("Data Balanced Sucessfully")
            return balanced_df
        
        except Exception as e:
            logger.info(f"Error During Handling im-balance data {e}")
            raise CustomException(f"Error While Handling im-balance data",e)
        
    def feature_selection(self,df):
        try:
            logger.info("Starting our Feature Selection Step")
            X = df.drop(columns='booking_status',axis=1)
            y=df['booking_status']
            model=RandomForestClassifier()
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importances= pd.DataFrame({'feature':X.columns,'importance':feature_importance})
            top_features_importance_df=feature_importances.sort_values(by='importance', ascending=False)

            num_features_to_select = self.config['data_processing']['no_of_features']
            top_ten_features = top_features_importance_df['feature'].head(num_features_to_select).values
            logger.info(f"Feature Selected : {top_ten_features}")
            top_10_df = df[top_ten_features.tolist() + ['booking_status']]


            logger.info("Feature Selection completed sucessfully")

            return top_10_df
        
        except Exception as e:
            logger.info(f"Error During Feature Selection {e}")
            raise CustomException(f"Error While Feature Selection",e)
        
        
    def save_data(self, df, file_path):
        try:
            logger.info("saving our data in processed folder")

            df.to_csv(file_path, index=False)


            logger.info(f"Data saved sucessfully to {file_path}")
        except Exception as e:
            logger.info(f"Error During saving data {e}")
            raise CustomException(f"Error While saving data",e)
    
    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)


            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)


            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]


            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing completed sucessfully")
        except Exception as e:
            logger.info(f"Error During preprocessing pipeline {e}")
            raise CustomException(f"Error While Data Preprocessing pipeline",e)
        

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()




