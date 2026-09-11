"""
model_configs.py

Plain scikit-learn / xgboost model instances and small hyperparameter
grids for GridSearchCV. No custom wrapper classes - these are used
directly inside a Pipeline's final step, which is the standard pattern.

Grids are intentionally small. The brief is explicit about this: don't run
an enormous hyperparameter search on a small biological dataset, a
reasonable grid searched properly beats a huge grid searched on too little
data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models_and_grids(random_state: int = 42) -> dict:
    """Returns {model_name: (estimator, param_grid)}. Param grid keys are
    prefixed with 'clf__' so they work directly with a Pipeline named step
    called 'clf' - see run_nested_cv.py."""

    elastic_net = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=5000, random_state=random_state
    )
    elastic_net_grid = {
        "clf__C": [0.01, 0.1, 1.0],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
    }

    random_forest = RandomForestClassifier(random_state=random_state)
    random_forest_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3],
    }

    xgboost_model = XGBClassifier(
        eval_metric="logloss", random_state=random_state
    )
    xgboost_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
    }

    return {
        "elastic_net": (elastic_net, elastic_net_grid),
        "random_forest": (random_forest, random_forest_grid),
        "xgboost": (xgboost_model, xgboost_grid),
    }
