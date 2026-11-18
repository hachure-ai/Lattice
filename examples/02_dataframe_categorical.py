"""pandas input with categorical columns and missing values.

Requires pandas:
    pip install pandas
    python examples/02_dataframe_categorical.py

For DataFrames the column dtypes decide which features are categorical, so no
``categorical_indices`` are needed. Lattice fills numeric gaps from the support
mean and adds missing-value indicators during preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from lattice import LatticeClassifier

FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]
REGIONS = ["north", "south", "east", "west"]


def make_frame(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(X, columns=FEATURES)
    labels = [REGIONS[i % 3] for i in range(len(frame))]
    frame["region"] = pd.Categorical(labels, categories=REGIONS)
    frame["target"] = y
    return frame


def main() -> None:
    X, y = load_wine(return_X_y=True)
    frame = make_frame(X, y)
    train, test = train_test_split(frame, test_size=0.25, random_state=0, stratify=frame["target"])

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    # Simulate a partially observed feature and an unseen category.
    X_test = X_test.copy()
    X_test.loc[X_test.index[:5], "alcohol"] = np.nan
    X_test.loc[X_test.index[-3:], "region"] = "west"

    clf = LatticeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    result = X_test[["region"]].copy()
    result["predicted"] = predictions
    result["actual"] = y_test.to_numpy()
    print(f"accuracy with missing values: {(predictions == y_test.to_numpy()).mean():.3f}")
    print(result.head(10).to_string())


if __name__ == "__main__":
    main()
