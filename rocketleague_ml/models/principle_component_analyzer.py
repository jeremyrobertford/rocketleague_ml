import os
import pandas as pd
import numpy as np
from rocketleague_ml.config import FEATURES
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo  # type: ignore
from factor_analyzer.rotator import Rotator  # pyright: ignore[reportMissingTypeStubs]
from scipy.stats import skew  # type: ignore
from typing import cast, List, Tuple, TypedDict, Dict, Any


class FeatureComparison(TypedDict):
    feature: str
    skew_before: float
    skew_after: float
    transform: str


class Principle_Component_Analyzer:
    def __init__(self):
        pass

    def transform_skewed_features(
        self, features: pd.DataFrame, zero_heavy: List[str] | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform numeric features to reduce skew for PCA.

        Parameters
        ----------
        features : pd.DataFrame
            Input feature DataFrame
        zero_heavy : list, optional
            List of columns with extreme zero-heavy distributions for stronger transform

        Returns
        -------
        skewed_features : pd.DataFrame
            DataFrame with transformed numeric features
        skew_compare : pd.DataFrame
            Table with skew before/after and applied transform
        """
        if zero_heavy is None:
            zero_heavy = [
                "Boost Efficiency",
                "Far From Ball Boost Efficiency",
                "Percent Time while Slow",
            ]

        skewed_features = features.copy()
        records: List[FeatureComparison] = []

        for col in skewed_features.select_dtypes(include=["float", "int"]).columns:
            data = skewed_features[col].dropna()

            if len(data) == 0:
                continue

            s_before = skew(data)
            if np.isnan(s_before) or abs(s_before) < 0.5:
                continue

            # Special-case zero-heavy columns
            if col in zero_heavy:
                x: np.ndarray[Tuple[int, int], np.dtype[np.float64]] = np.log1p(data * 100)  # type: ignore
                pt = PowerTransformer(method="yeo-johnson")
                flattened_x = cast(np.ndarray[Tuple[int], np.dtype[np.float64]], x.values.reshape(-1, 1))  # type: ignore
                transformed = cast(
                    np.ndarray[Tuple[int], np.dtype[np.float64]],
                    pt.fit_transform(flattened_x).flatten(),  # type: ignore
                )
                method = "log1p+yeo-johnson"

            else:
                # Regular skew handling
                if s_before > 1:
                    if (data >= 0).all() and (data == 0).any():
                        flattened_x = cast(np.ndarray[Tuple[int], np.dtype[np.float64]], data.values.reshape(-1, 1))  # type: ignore
                        pt = PowerTransformer(method="yeo-johnson")
                        transformed = cast(
                            np.ndarray[Tuple[int], np.dtype[np.float64]],
                            pt.fit_transform(flattened_x).flatten(),  # type: ignore
                        )
                        method = "yeo-johnson"
                    else:
                        transformed = np.log1p(data)
                        method = "log1p"
                elif s_before > 0.5:
                    transformed = np.sqrt(data)
                    method = "sqrt"
                elif s_before < -0.5:
                    transformed = np.power(data, 2)
                    method = "square"
                else:
                    transformed = data.copy()
                    method = "none"

            skewed_features[col] = transformed
            s_after = skew(transformed)
            records.append(
                {
                    "feature": col,
                    "skew_before": s_before,
                    "skew_after": s_after,
                    "transform": method,
                }
            )

        skew_compare = pd.DataFrame(records)
        skew_compare.sort_values("skew_after", ascending=False, inplace=True)  # type: ignore
        skew_compare.reset_index(drop=True, inplace=True)
        return skewed_features, skew_compare

    def drop_features(self, features: pd.DataFrame):
        # Define spatial keywords by dimension
        field_zones = ["Offensive", "Neutral", "Defensive"]
        field_lateral = ["Left", "Middle", "Right"]
        field_height = ["Lowest", "Middle-Aerial", "Highest"]

        def count_keywords(name: str, keywords: List[str]):
            return sum(1 for k in keywords if k in name)

        def is_nested_spatial_col(name: str):
            # Count how many distinct dimensions appear
            zone_hits = count_keywords(name, field_zones)
            lat_hits = count_keywords(name, field_lateral)
            height_hits = count_keywords(name, field_height)
            total_hits = zone_hits + lat_hits + height_hits

            # Drop if column name contains 2+ of these components
            return (
                total_hits >= 2 and "Third" in name
            )  # only apply to positional "Third" metrics

        drop_nested_spatial = [c for c in features.columns if is_nested_spatial_col(c)]

        drop_features = [
            # --- Co-dependent columsn ---
            *[
                c
                for c in features.columns
                if "Middle" in c
                or "Neutral" in c
                or "Behind Ball" in c
                or "Airborne" in c
                or "Medium-Speed" in c
                or ">50" in c
                or "Second Man" in c
                or "2 to 1" in c
            ],
            # --- Redundant "Average Stint ..." versions (keep only Percent Time) ---
            *[c for c in features.columns if "Average Stint" in c],
            # --- Team-level or opponent-level fields ---
            *[
                c
                for c in features.columns
                if any(
                    x in c
                    for x in [
                        "Team in",
                        "Team With",
                        "Team with",
                        "Opponent Team",
                        "Percentage Blue",
                        "Percentage Orange",
                    ]
                )
            ],
            # --- Aggregated spatial layers (too coarse or sums of others) ---
            *[c for c in features.columns if any(x in c for x in ["Half"])],
            # --- Nested redundant thirds (we keep basic Third layers, not all cross-combos) ---
            *drop_nested_spatial,
            # --- Rotations: drop one full system (we’ll keep "Full" and drop "Simple") ---
            *[c for c in features.columns if "Simple" in c and "Stolen" not in c],
            # --- Redundant Boost Efficiency subtypes (keep just "Boost Efficiency" and maybe "Far From Ball") ---
            *[
                c
                for c in features.columns
                if any(
                    x in c
                    for x in [
                        "Supersonic Speed Boost Efficiency",
                        "Drive to Boost Speed Boost Efficiency",
                        "Simple Boost Efficiency",
                    ]
                )
            ],
            # --- Scored/outcome metrics (not behavior) ---
            "Scored Goal",
            "Team Scored Goal",
            # --- Categorical: drop bins and use ordinal version ---
            *[
                c
                for c in features.columns
                if ">0" in c
                or ">25" in c
                or ">75" in c
                or "Full Boost" in c
                or "No Boost" in c
                or "Slow" in c
                or "Fast" in c
                or "Drive Speed" in c
                or "Boost Speed" in c
                or "Rotating From" in c
            ],
            "Percent Time while Supersonic",
        ]
        features = features.drop(columns=drop_features, errors="ignore")
        return features

    def perform_factor_analysis(
        self,
        features: pd.DataFrame,
        n_factors: int | None = None,
        rotation: str = "varimax",
        min_eigenvalue: float = 1.0,
    ):
        """
        Perform exploratory factor analysis (EFA) on a dataset.

        Parameters
        ----------
        features : pd.DataFrame
            Input features (numeric only).
        n_factors : int, optional
            Number of factors to extract. If None, determined via eigenvalues > min_eigenvalue.
        rotation : str, default='varimax'
            Rotation method ('varimax', 'promax', etc.).
        min_eigenvalue : float, default=1.0
            Minimum eigenvalue threshold for factor retention (Kaiser criterion).
        return_full : bool, default=False
            If True, return full analysis dictionary; else return just loadings.

        Returns
        -------
        pd.DataFrame or dict
            Factor loadings, or full result dict with loadings, communalities, eigenvalues, and variance explained.
        """

        # 1. Standardize data
        X = features

        # 2. Test suitability (optional but good sanity check)
        _, bartlett_p = calculate_bartlett_sphericity(X)
        _, kmo_model = calculate_kmo(X)  # type: ignore

        # 3. Compute eigenvalues to decide number of factors if needed
        fa_temp = FactorAnalyzer(rotation=rotation)
        for col in X.columns:
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                print(f"NaN counts: {col} {nan_count}")
            inf_count = np.isinf(X[col]).sum()
            if inf_count > 0:
                print(f"Inf counts: {col} {inf_count}")
        fa_temp.fit(X)  # type: ignore
        eigenvalues, _ = fa_temp.get_eigenvalues()

        if n_factors is None:
            n_f: int = np.sum(eigenvalues > min_eigenvalue)  # type: ignore
        else:
            n_f = n_factors

        # 4. Fit final factor model
        fa = FactorAnalyzer(n_factors=n_f, rotation=rotation)
        fa.fit(X)  # type: ignore

        # 5. Collect results
        loadings = pd.DataFrame(
            fa.loadings_,  # type: ignore
            index=features.columns,
            columns=[f"Factor{i+1}" for i in range(n_f)],
        )
        loadings.to_csv(os.path.join(FEATURES, "debug_fa_loadings.csv"))

        communalities = pd.Series(
            fa.get_communalities(), index=features.columns, name="Communality"  # type: ignore
        )
        variance = pd.DataFrame(
            {
                "SS Loadings": fa.get_factor_variance()[0],
                "Proportion Var": fa.get_factor_variance()[1],
                "Cumulative Var": fa.get_factor_variance()[2],
            },
            index=[f"Factor{i+1}" for i in range(n_f)],
        )

        results: Dict[str, Any] = {  # type: ignore
            "loadings": loadings,
            "communalities": communalities,
            "eigenvalues": eigenvalues,
            "variance": variance,
            "bartlett_p": bartlett_p,
            "kmo": kmo_model,
        }

        return loadings

    def analyze(self, features: pd.DataFrame):
        # 1. Summary stats
        features = self.drop_features(features)
        features = features.loc[:, features.nunique() > 1]
        features.describe().to_csv(os.path.join(FEATURES, "features_describe.csv"))

        # 2. Transform skewed features (assuming you have this method defined elsewhere)
        features, _ = self.transform_skewed_features(features=features)

        # 3. Select numeric columns
        numeric_cols = features.select_dtypes(include=["float", "int"]).columns

        # 4. Handle missing values
        features.fillna(0, inplace=True)  # type: ignore

        # 5. Run PCA
        n_features = 14  # or features.shape[1] if you want full dimensionality
        pca = PCA(n_components=n_features)
        pca.fit(features)  # type: ignore

        # 6. Explained variance
        explained_variance = pca.explained_variance_ratio_  # type: ignore
        cumulative_variance = np.cumsum(explained_variance)  # type: ignore
        pca_variance_df = pd.DataFrame(
            {
                "PC": range(1, n_features + 1),
                "ExplainedVariance": explained_variance,
                "CumulativeVariance": cumulative_variance,
            }
        )
        pca_variance_df.to_csv(os.path.join(FEATURES, "debug_pca.csv"), index=False)

        # 7. Loadings matrix
        loadings = pd.DataFrame(
            pca.components_.T,  # type: ignore
            index=numeric_cols,
            columns=[f"PC{i+1}" for i in range(n_features)],
        )
        loadings.to_csv(os.path.join(FEATURES, "debug_pca_loadings.csv"))

        # 8. Varimax rotation
        varimax_rotator = Rotator(method="varimax")
        rotated_varimax = varimax_rotator.fit_transform(loadings.values.T).T  # type: ignore
        rotated_varimax_df = pd.DataFrame(
            rotated_varimax,
            index=numeric_cols,
            columns=[f"PC{i+1}" for i in range(n_features)],
        )
        varimax_path = os.path.join(FEATURES, "debug_rotated_pca_loadings_varimax.csv")
        rotated_varimax_df.to_csv(varimax_path)

        # 9. Promax rotation
        # Try Promax rotation safely
        try:
            promax_rotator = Rotator(method="promax")
            rotated_promax = promax_rotator.fit_transform(loadings.values.T).T  # type: ignore
            rotated_promax_df = pd.DataFrame(
                rotated_promax,
                index=numeric_cols,
                columns=[f"PC{i+1}" for i in range(n_features)],
            )
            promax_path = os.path.join(
                FEATURES, "debug_rotated_pca_loadings_promax.csv"
            )
            rotated_promax_df.to_csv(promax_path)
        except np.linalg.LinAlgError:
            pass

        self.perform_factor_analysis(features, n_factors=n_features)

        return None
