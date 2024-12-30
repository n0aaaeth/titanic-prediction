import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm
from lightgbm import LGBMClassifier
import pickle
import warnings

warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

class GenerateModel:

    base_dir = Path(__file__).resolve().parent

    def __init__(self):
        pass

    @classmethod
    def __read_data(cls) -> pd.DataFrame:
        path_to_data = cls.base_dir / 'data' / 'train.csv'
        print(f"Loading data from: {path_to_data}")
        try:
            df = pd.read_csv(path_to_data)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {path_to_data}")
            raise

    @staticmethod
    def __extract_required_columns(df: pd.DataFrame):
        print("Extracting required columns...")
        required_columns = ['Survived', 'Sex', 'Pclass', 'Age', 'Parch', 'SibSp']
        try:
            df = df[required_columns]
            print(f"Columns extracted successfully. Shape: {df.shape}")
            return df
        except KeyError as e:
            print(f"Error: Missing required columns. {e}")
            raise

    @staticmethod
    def encode_sex(x: str):
        if x in ('male', '男性'):
            return 1
        elif x in ('female', '女性'):
            return 0
        else:
            return np.nan

    @classmethod
    def __preprocess_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing data...")
        tmp_df = cls.__extract_required_columns(df)
        tmp_df['Sex'] = tmp_df['Sex'].apply(lambda x: cls.encode_sex(x))
        tmp_df['Age'] = tmp_df['Age'].fillna(tmp_df['Age'].median())
        print(f"Data preprocessing completed. Shape: {tmp_df.shape}")
        return tmp_df

    @staticmethod
    def __train_model(df: pd.DataFrame) -> lightgbm.sklearn.LGBMModel:
        print("Training the model...")
        y = df['Survived']
        X = df.drop(['Survived'], axis=1)
        model = LGBMClassifier()
        model.fit(X.values, y.values)
        print("Model training completed.")
        return model

    @classmethod
    def __save_model(cls, model: lightgbm.sklearn.LGBMModel):
        path_to_model = cls.base_dir / 'model' / 'model.pkl'
        print(f"Saving model to: {path_to_model}")
        try:
            with open(path_to_model, "wb") as f:
                pickle.dump(model, f)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error while saving model: {e}")
            raise

    @classmethod
    def generate_model(cls):
        print("Starting model generation...")
        try:
            df = cls.__read_data()
            preprocessed_df = cls.__preprocess_df(df)
            lgbm_model = cls.__train_model(preprocessed_df)
            cls.__save_model(lgbm_model)
            print("Model generation process completed successfully!")
        except Exception as e:
            print(f"Error during model generation: {e}")
            raise


class PredictOnAPI(GenerateModel):

    def __init__(self):
        pass

    @classmethod
    def __load_model(cls):
        path_to_model = cls.base_dir / 'model' / 'model.pkl'
        if not path_to_model.exists():
            print("Model file not found. Generating a new model...")
            cls.generate_model()

        try:
            with open(path_to_model, "rb") as f:
                model = pickle.load(f)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error while loading model: {e}")
            raise

    @staticmethod
    def __encode_pclass(x: str):
        if x == '上層クラス（お金持ち）':
            return 1
        elif x == '中級クラス（一般階級）':
            return 2
        elif x == '下層クラス（労働階級）':
            return 3
        else:
            return np.nan

    @classmethod
    def derive_survival_probability(
        cls,
        Sex: str,
        Pclass: str,
        Age: int,
        Parch: int,
        SibSp: int
    ) -> float:
        print("Predicting survival probability...")
        model = cls.__load_model()
        encoded_sex = cls.encode_sex(Sex)
        encoded_pclass = cls.__encode_pclass(Pclass)

        features = np.array([[
            encoded_sex, encoded_pclass, Age, Parch, SibSp
        ]])

        survival_probability = model.predict_proba(features)[0][1]
        print(f"Survival probability calculated: {survival_probability}")
        return round(survival_probability, 3)


if __name__ == '__main__':
    GenerateModel.generate_model()
