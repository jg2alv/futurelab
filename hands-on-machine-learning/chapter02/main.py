import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.impute
import numpy as np

def load_housing_data():
    return pd.read_csv(pathlib.Path("data/housing.csv"))

def main():
    df = load_housing_data()
    
    df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1,2,3,4,5])
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42, stratify=df["income_cat"])
    train.drop("income_cat", axis=1, inplace=True)
    test.drop("income_cat", axis=1, inplace=True)

    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    imputer = sklearn.impute.SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    print(housing[["ocean_proximity"]])

if __name__ == '__main__':
    main()