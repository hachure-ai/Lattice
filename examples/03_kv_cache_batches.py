"""Reuse one support-set encoding across many prediction calls.

    python examples/03_kv_cache_batches.py

With ``kv_cache=True`` the support set is encoded once and reused, which pays
off when the same fitted estimator scores many incoming batches.
"""

import time

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lattice import LatticeRegressor


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    reg = LatticeRegressor(kv_cache=True, kv_cache_device="auto")
    reg.fit(X_train, y_train)

    batches = np.array_split(X_test, 4)
    started = time.perf_counter()
    for index, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        predictions = reg.predict(batch)
        print(
            f"batch {index}: rows={len(batch)} "
            f"seconds={time.perf_counter() - batch_started:.3f} "
            f"mean={predictions.mean():.1f}"
        )
    print(f"total seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
