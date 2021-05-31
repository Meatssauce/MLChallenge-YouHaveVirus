from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import xgboost as xgb

from tools import *

# from word2number import w2n

from sklearn.base import BaseEstimator, TransformerMixin
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


def main():
    # Parameters
    seed = np.random.seed(0)
    target_feature = 'Prediction'

    # Make a composite estimator that includes preprocessing
    pipe = make_pipeline(
        Preprocessor(bins=14),
        FeatureUnion(transformer_list=[
            ("numeric_features", make_pipeline(
                TypeSelector(np.number),
                SimpleImputer(strategy="median"),
                StandardScaler()
            )),
            ("categorical_features", make_pipeline(
                TypeSelector(object), # TypeSelector("category"),
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder()
            )),
            ("boolean_features", make_pipeline(
                TypeSelector("bool"),
                SimpleImputer(strategy="most_frequent")
            ))
        ]),
        xgb.XGBClassifier(objective ='binary:logistic', random_state=seed, verbosity=0, scoring='f1', n_jobs=-1)
    )

    df = load_data(labels_path='dataset/train.csv', conversations_folder_path='dataset/trainConversations', verbose=1)

    # drop duplicates, empty rows and columns and rows with invalid labels
    df.dropna(axis=0, how='any', subset=[target_feature], inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    temp = pd.read_csv('dataset/generated/transformed_train.csv') # 'dataset/generated/transformed_train.csv'
    temp = temp.rename(columns={'Prediction': 'GPT2_Prediction'})
    df.join(temp.set_index('ID'), on='ID')

    X, y = df.drop(columns=[target_feature]), df[target_feature].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    pipe.fit(X_train, y_train)

    df_pred = load_data(labels_path='dataset/test.csv', conversations_folder_path='dataset/testConversations', verbose=1)
    df_pred = pipe.transform(df_pred)
    y_pred = pipe.predict(df_pred)
    y_pred = pd.DataFrame(y_pred, columns=[target_feature])

    prediction = pd.concat([df_pred['ID'], y_pred], axis=1)
    prediction.to_csv('dataset/generated/prediction2.csv', index=False)

if __name__ == '__main__':
    main()