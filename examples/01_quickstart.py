"""Classification and regression in one script.

Run with:
    python examples/01_quickstart.py

The released checkpoints are downloaded from the Hugging Face Hub on first use
and cached by ``huggingface_hub``.
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from lattice import LatticeClassifier, LatticeRegressor


def classification() -> None:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    clf = LatticeClassifier()
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)
    predictions = clf.predict(X_test)

    print(f"[classification] accuracy={accuracy_score(y_test, predictions):.3f}")
    print(f"[classification] classes={clf.classes_.tolist()}")
    print(f"[classification] probabilities: {probabilities.shape}")


def regression() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    reg = LatticeRegressor()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    print(f"[regression] r2={r2_score(y_test, predictions):.3f} rmse={rmse:.1f}")


if __name__ == "__main__":
    classification()
    regression()
