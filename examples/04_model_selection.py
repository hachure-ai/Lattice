"""Tune the ensemble size with scikit-learn's GridSearchCV.

    python examples/04_model_selection.py
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from lattice import LatticeClassifier


def main() -> None:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    search = GridSearchCV(
        LatticeClassifier(),
        {"n_estimators": [4, 8], "batch_size": [4, 8]},
        scoring="accuracy",
        cv=3,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    print(f"best params: {search.best_params_}")
    print(f"best cross-validation accuracy: {search.best_score_:.3f}")
    predictions = search.predict(X_test)
    print(f"held-out accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"per-fold scores: {np.round(search.cv_results_['mean_test_score'], 3).tolist()}")


if __name__ == "__main__":
    main()
