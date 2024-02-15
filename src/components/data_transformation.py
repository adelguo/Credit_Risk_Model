from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
import os
import sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            catigorical_columns=['Sex','Job','Housing','Saving accounts','Checking account','Purpose']
            numerical_columns=['Age','Credit amount','Duration']

            num_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')),
                                    ('scaler', StandardScaler())])
            cat_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
                                                ('encoder', OneHotEncoder())])                       

            preprocessor = ColumnTransformer(
                [
                    ("catigorical", cat_transformer, catigorical_columns),
                    ('numerical', num_transformer, numerical_columns)
                ]
            )

            