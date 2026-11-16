"""
Classical baseline models — v8
================================
Literature-based classical methods for HSI classification:

1. GNB_HOSD      — Duan et al. 2023 original HOSD paper method
2. RandomForest  — standard ensemble baseline
3. SVM_RBF       — classic HSI classification (Melgani & Bruzzone 2004)
4. LDA           — Linear Discriminant Analysis (spectral baseline)
5. ExtraTrees    — extremely randomized trees (faster than RF)
6. GradientBoost — XGBoost-style gradient boosting

Literature references:
  - SVM: Melgani & Bruzzone, IEEE TGRS 2004 (classic HSI SVM)
  - RF:  Ham et al., IEEE TGRS 2005 (RF for HSI)
  - LDA: Bandos et al., IEEE TGRS 2009 (regularized LDA for HSI)
  - GNB: Duan et al., IEEE TGRS 2023 (HOSD original)
"""

import numpy as np
from sklearn.svm             import SVC
from sklearn.ensemble        import (RandomForestClassifier,
                                     ExtraTreesClassifier,
                                     GradientBoostingClassifier)
from sklearn.naive_bayes     import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from typing import Dict

from config import RF_N_EST, RF_MAX_DEPTH


# ──────────────────────────────────────────────────────────────
def get_classical_models(n_jobs: int = -1) -> Dict[str, object]:
    """
    Return dict of {name: sklearn-estimator} for all classical baselines.
    Each estimator exposes .fit() and .predict_proba().
    """
    models = {
        # ── Duan et al. 2023 — original HOSD baseline ─────────
        "GNB": GaussianNB(),

        # ── Random Forest — Ham et al. 2005 ───────────────────
        "RandomForest": RandomForestClassifier(
            n_estimators=RF_N_EST,
            max_depth=RF_MAX_DEPTH,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        ),

        # ── SVM RBF — Melgani & Bruzzone 2004 ─────────────────
        # "SVM_RBF": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("clf", SVC(
        #         kernel="rbf",
        #         C=SVM_C,
        #         gamma=SVM_GAMMA,
        #         probability=True,
        #         class_weight="balanced",
        #         random_state=42,
        #     )),
        # ]),

        # ── Linear Discriminant Analysis — Bandos et al. 2009 ─
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(
                solver="svd",
                tol=1e-4,
            )),
        ]),

        # ── Extra Trees — faster RF variant ───────────────────
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=RF_N_EST,
            max_depth=RF_MAX_DEPTH,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        ),
    }
    return models
