# Copyright 2025 RHoMIS Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Food Security Prediction Assistant - Chat-Based ML Deployment App (v3)

Aligned with food_security_notebook_v2_improved.ipynb:
  - 4-class ordinal target: FoodSecure, MildlyFI, ModeratelyFI, SeverelyFI
  - Primary metric: Weighted (Quadratic) Cohen's Kappa
  - Full ~20-feature input schema (leakage columns removed)
  - Serving-side feature engineering mirroring notebook sections 4.3-4.6
  - Dual preprocessing paths: sklearn ColumnTransformer vs CatBoost native
  - Stacking Ensemble as default model
  - SHAP local explanations (optional pane)
  - Per-class probability display
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import time
import traceback
from pathlib import Path
from collections import namedtuple

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Food Security Prediction Assistant",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

HFIAS_ORDER = ['FoodSecure', 'MildlyFI', 'ModeratelyFI', 'SeverelyFI']

CLASS_META = {
    'FoodSecure':   {'emoji': '✅', 'color': '#2e7d32', 'label': 'Food Secure',
                     'desc': 'Household meets food needs throughout the year.'},
    'MildlyFI':     {'emoji': '🟡', 'color': '#f9a825', 'label': 'Mildly Food Insecure',
                     'desc': 'Occasional anxiety about food access; quality/quantity not yet reduced.'},
    'ModeratelyFI': {'emoji': '🟠', 'color': '#e65100', 'label': 'Moderately Food Insecure',
                     'desc': 'Reduced quality or quantity of food consumed regularly.'},
    'SeverelyFI':   {'emoji': '🔴', 'color': '#b71c1c', 'label': 'Severely Food Insecure',
                     'desc': 'Severe food shortage; household experiencing hunger.'},
}

MODELS_DIR = Path("savedModels")

MODEL_INFO = {
    "Stacking Ensemble": {
        "file": MODELS_DIR / "stacking_ensemble.pkl",
        "icon": "🏆",
        "description": "Best model — LGBM + XGB + RF → LogReg meta-learner",
        "weighted_kappa": 0.6504,
        "roc_auc": 0.8285,
        "requires_catboost_path": False,
    },
    "LightGBM": {
        "file": MODELS_DIR / "lightgbm_tuned.pkl",
        "icon": "🌿",
        "description": "Best single model — fast leaf-wise boosting",
        "weighted_kappa": 0.6461,
        "roc_auc": 0.8251,
        "requires_catboost_path": False,
    },
    "Voting Ensemble": {
        "file": MODELS_DIR / "voting_ensemble.pkl",
        "icon": "🗳️",
        "description": "Soft-voting — LGBM + XGB + RF (best ROC-AUC)",
        "weighted_kappa": 0.6449,
        "roc_auc": 0.8294,
        "requires_catboost_path": False,
    },
    "XGBoost": {
        "file": MODELS_DIR / "xgboost_tuned.pkl",
        "icon": "🍃",
        "description": "Optuna-tuned gradient boosting",
        "weighted_kappa": 0.6327,
        "roc_auc": 0.8189,
        "requires_catboost_path": False,
    },
    "Random Forest": {
        "file": MODELS_DIR / "random_forest.pkl",
        "icon": "🌲",
        "description": "Balanced bagging ensemble",
        "weighted_kappa": 0.6192,
        "roc_auc": 0.8121,
        "requires_catboost_path": False,
    },
    "CatBoost": {
        "file": MODELS_DIR / "catboost_tuned.pkl",
        "icon": "🐱",
        "description": "Native categorical handling, Optuna-tuned",
        "weighted_kappa": 0.6187,
        "roc_auc": 0.8109,
        "requires_catboost_path": True,
    },
}

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES = [
    'HHsizeMAE',
    'LandOwned',
    'LandCultivated',
    'LivestockHoldings',
    'PPI_Likelihood',
    'TVA_USD_PPP_pmae_pday',
    'total_income_USD_PPP_pHH_Yr',
    'offfarm_income_USD_PPP_pHH_Yr',
    'farm_income_USD_PPP_pHH_Yr',
    'value_crop_consumed_USD_PPP_pHH_Yr',
    'livestock_prodsales_USD_PPP_pHH_Yr',
    'value_livestock_production_USD_PPP_pHH_Yr',
    'Food_Self_Sufficiency_kCal_MAE_day',
    'score_HDDS_GoodSeason',
    'score_HDDS_farmbasedGoodSeason',
    'score_HDDS_purchasedGoodSeason',
    'score_HDDS_BadSeason',
    'score_HDDS_farmbasedBadSeason',
    'score_HDDS_purchasedBadSeason',
    'NrofMonthsWildFoodCons',
    'GHGEmissions',
    'NFertInput',
    'Altitude',
    'Gender_FemaleControl',
    'Market_Orientation',
]

CATEGORICAL_FEATURES = {
    'Country': [
        'Tanzania', 'Burkina_Faso', 'Cambodia', 'DRC', 'Ethiopia',
        'Ghana', 'India', 'Kenya', 'LaoPDR', 'Malawi', 'Mali',
        'Nicaragua', 'Peru', 'Uganda', 'Vietnam', 'Zambia',
        'Burundi', 'El_Salvador', 'Costa_Rica', 'Guatemala', 'Honduras',
    ],
    'HouseholdType': ['couple', 'single', 'polygamous', 'other'],
    'Head_EducationLevel': [
        'no_school', 'primary', 'secondary', 'postsecondary',
        'adult_education', 'literate', 'religious_school', 'other',
    ],
    'Livestock_Orientation': ['subsistence', 'mixed', 'commercial', 'other'],
}

OPTIONAL_FEATURES = {
    'GHGEmissions', 'NFertInput', 'Altitude',
    'NrofMonthsWildFoodCons',
    'value_crop_consumed_USD_PPP_pHH_Yr',
    'value_livestock_production_USD_PPP_pHH_Yr',
    'livestock_prodsales_USD_PPP_pHH_Yr',
    'offfarm_income_USD_PPP_pHH_Yr',
    'score_HDDS_farmbasedGoodSeason',
    'score_HDDS_purchasedGoodSeason',
    'score_HDDS_farmbasedBadSeason',
    'score_HDDS_purchasedBadSeason',
    'Market_Orientation',
    'Head_EducationLevel',
    'HouseholdType',
    'Livestock_Orientation',
}

FEATURE_QUESTIONS = {
    'Country':                                   "Let's start — which country is your household located in?",
    'HHsizeMAE':                                 "How many adult-equivalent household members are there? (1 adult=1.0, 1 child≈0.5)",
    'LandOwned':                                 "How much land does your household own? (hectares; 1 ha ≈ 2.5 acres)",
    'LandCultivated':                            "How much land does your household actively cultivate? (hectares)",
    'LivestockHoldings':                         "Total livestock in Tropical Livestock Units (TLU)? (1 cattle=1.0, 1 goat=0.1, 10 chickens=0.1)",
    'PPI_Likelihood':                            "Poverty probability index — likelihood the household is above the poverty line (0–100)",
    'TVA_USD_PPP_pmae_pday':                     "Total value of all activities per adult-equivalent per day (USD PPP)",
    'total_income_USD_PPP_pHH_Yr':               "Total household income per year (USD PPP)",
    'offfarm_income_USD_PPP_pHH_Yr':             "(Optional) Off-farm income per year (USD PPP; enter 0 if none)",
    'farm_income_USD_PPP_pHH_Yr':                "Farm income per year (USD PPP)",
    'value_crop_consumed_USD_PPP_pHH_Yr':        "(Optional) Value of crops consumed at home per year (USD PPP)",
    'livestock_prodsales_USD_PPP_pHH_Yr':        "(Optional) Value of livestock products sold per year (USD PPP)",
    'value_livestock_production_USD_PPP_pHH_Yr': "(Optional) Total value of livestock production per year (USD PPP)",
    'Food_Self_Sufficiency_kCal_MAE_day':        "Food self-sufficiency — calories produced per adult-equivalent per day (kcal)",
    'score_HDDS_GoodSeason':                     "Dietary Diversity Score in the GOOD season (0–12 food groups)",
    'score_HDDS_farmbasedGoodSeason':            "(Optional) Farm-based food groups in the good season (0–12)",
    'score_HDDS_purchasedGoodSeason':            "(Optional) Purchased food groups in the good season (0–12)",
    'score_HDDS_BadSeason':                      "Dietary Diversity Score in the BAD/lean season (0–12 food groups)",
    'score_HDDS_farmbasedBadSeason':             "(Optional) Farm-based food groups in the lean season (0–12)",
    'score_HDDS_purchasedBadSeason':             "(Optional) Purchased food groups in the lean season (0–12)",
    'NrofMonthsWildFoodCons':                    "(Optional) Months per year the household consumes wild foods (0–12)",
    'GHGEmissions':                              "(Optional) GHG emissions estimate for the farm (kg CO2-eq/year)",
    'NFertInput':                                "(Optional) Nitrogen fertiliser input (kg/year)",
    'Altitude':                                  "(Optional) Altitude of the household location (metres above sea level)",
    'Gender_FemaleControl':                      "Proportion of household decisions made by women (0=none, 0.5=half, 1=all)",
    'Market_Orientation':                        "(Optional) Share of farm output sold on markets (0–1)",
    'HouseholdType':                             "(Optional) Household composition type",
    'Head_EducationLevel':                       "(Optional) Education level of the household head",
    'Livestock_Orientation':                     "(Optional) Primary livestock management orientation",
}

FEATURE_DESCRIPTIONS = {
    'HHsizeMAE':                        '💡 Count all family members, adjusting for age: adults=1.0, children=0.5',
    'LandOwned':                        '💡 Include all land owned or controlled by the household',
    'LandCultivated':                   '💡 Only land actively farmed this season',
    'LivestockHoldings':                '💡 TLU equivalents: 1 cattle=1.0, 1 sheep/goat=0.1, 10 chickens≈0.1',
    'PPI_Likelihood':                   '💡 Higher = more likely above poverty line. Use your best estimate (0–100)',
    'score_HDDS_GoodSeason':            '💡 Count distinct food groups: grains, legumes, vegetables, fruits, meat, fish, eggs, dairy, oils, sweets, condiments',
    'score_HDDS_BadSeason':             '💡 Same food groups but during the lean/dry season — the most predictive single feature in the model',
    'Food_Self_Sufficiency_kCal_MAE_day': '💡 Subsistence ≈ 500–1500 kcal/MAE/day; moderate ≈ 1500–2500; surplus > 2500',
    'farm_income_USD_PPP_pHH_Yr':       '💡 Rough ranges: low < $300, medium $300–$1500, high > $1500 USD PPP/year',
    'total_income_USD_PPP_pHH_Yr':      '💡 Include all sources: crops, livestock, off-farm wages, remittances',
    'TVA_USD_PPP_pmae_pday':            '💡 Approximate: total annual income ÷ 365 ÷ household MAE size',
    'Gender_FemaleControl':             '💡 0 = all decisions by men, 0.5 = equal, 1 = all decisions by women',
}

FEATURE_BLOCKS = {
    "📍 Location":                ['Country'],
    "👨‍👩‍👧 Household":            ['HHsizeMAE', 'HouseholdType', 'Head_EducationLevel', 'Gender_FemaleControl'],
    "🌾 Land & Livestock":        ['LandOwned', 'LandCultivated', 'LivestockHoldings', 'Livestock_Orientation'],
    "🥗 Nutrition (Good Season)": ['score_HDDS_GoodSeason', 'score_HDDS_farmbasedGoodSeason',
                                   'score_HDDS_purchasedGoodSeason', 'Food_Self_Sufficiency_kCal_MAE_day'],
    "🥗 Nutrition (Lean Season)": ['score_HDDS_BadSeason', 'score_HDDS_farmbasedBadSeason',
                                   'score_HDDS_purchasedBadSeason', 'NrofMonthsWildFoodCons'],
    "💰 Income & Livelihoods":    ['total_income_USD_PPP_pHH_Yr', 'farm_income_USD_PPP_pHH_Yr',
                                   'offfarm_income_USD_PPP_pHH_Yr', 'TVA_USD_PPP_pmae_pday',
                                   'value_crop_consumed_USD_PPP_pHH_Yr',
                                   'livestock_prodsales_USD_PPP_pHH_Yr',
                                   'value_livestock_production_USD_PPP_pHH_Yr',
                                   'Market_Orientation'],
    "📊 Assets & Poverty":        ['PPI_Likelihood'],
    "🌍 Environment":             ['GHGEmissions', 'NFertInput', 'Altitude'],
}

MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
TaskInfo = namedtuple("TaskInfo", ["name", "value"])

EDU_MAP = {
    'no_school': 'no_school', 'no school': 'no_school',
    'illiterate': 'no_school', 'none': 'no_school',
    'adult_education': 'adult_education',
    'adult education, literacy school or parish school': 'adult_education',
    'postsecondary': 'postsecondary', 'post-secondary': 'postsecondary',
    'primary': 'primary', 'secondary': 'secondary',
    'literate': 'literate', 'religious_school': 'religious_school',
}
HT_MAP = {
    'couple': 'couple', 'together': 'couple',
    'couple_man_works_away': 'couple', 'couple_woman_works_away': 'couple',
    'woman_single': 'single', 'man_single': 'single', 'single': 'single',
    'polygamous': 'polygamous',
}
LOG_COLS = [
    'LandOwned', 'LandCultivated', 'LivestockHoldings',
    'TVA_USD_PPP_pmae_pday', 'total_income_USD_PPP_pHH_Yr',
    'offfarm_income_USD_PPP_pHH_Yr', 'farm_income_USD_PPP_pHH_Yr',
    'value_crop_consumed_USD_PPP_pHH_Yr',
    'value_livestock_production_USD_PPP_pHH_Yr',
    'livestock_prodsales_USD_PPP_pHH_Yr',
    'Food_Self_Sufficiency_kCal_MAE_day',
    'GHGEmissions', 'NFertInput', 'Altitude',
]
EPS = 1e-6


# ==============================================================================
# ARTIFACT LOADING
# ==============================================================================

@st.cache_resource(ttl=3600)
def load_artifacts():
    models = {}
    for model_name, info in MODEL_INFO.items():
        path = Path(info["file"])
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Could not load {model_name}: {e}")
                models[model_name] = None
        else:
            models[model_name] = None

    def _load(rel_path, label):
        p = MODELS_DIR / rel_path
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Could not load {label}: {e}")
        else:
            st.warning(f"⚠️ {label} not found at {p}")
        return None

    preprocessor  = _load("preprocessor.pkl",    "Preprocessor")
    num_imputer   = _load("num_imputer.pkl",      "Numeric imputer (CatBoost path)")
    label_encoder = _load("label_encoder.pkl",    "Label encoder")
    feature_meta  = _load("feature_metadata.pkl", "Feature metadata")

    return models, preprocessor, num_imputer, label_encoder, feature_meta


# ==============================================================================
# SERVING-SIDE FEATURE ENGINEERING  (mirrors notebook sections 4.3-4.6)
# ==============================================================================

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Section 4.3 — Standardise categoricals
    if 'Head_EducationLevel' in df.columns:
        df['Head_EducationLevel'] = (
            df['Head_EducationLevel'].astype(str).str.lower().str.strip()
            .map(EDU_MAP).fillna('other')
        )
    if 'HouseholdType' in df.columns:
        df['HouseholdType'] = (
            df['HouseholdType'].astype(str).str.lower().str.strip()
            .map(HT_MAP).fillna('other')
        )

    # Section 4.4 — log1p transforms & missingness flags
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(pd.to_numeric(df[col], errors='coerce').clip(lower=0))

    df['ghg_observed']   = (~df['GHGEmissions'].isna()).astype(int) if 'GHGEmissions' in df.columns else 0
    df['nfert_observed'] = (~df['NFertInput'].isna()).astype(int)   if 'NFertInput'   in df.columns else 0

    # Section 4.5 — Domain-specific features
    hhsize = df.get('HHsizeMAE', pd.Series([1.0])).fillna(1.0)
    if 'total_income_USD_PPP_pHH_Yr' in df.columns:
        df['income_per_MAE'] = df['total_income_USD_PPP_pHH_Yr'] / (hhsize + EPS)
    if 'LandCultivated' in df.columns:
        df['land_per_MAE'] = df['LandCultivated'] / (hhsize + EPS)
    if 'LivestockHoldings' in df.columns:
        df['livestock_per_MAE'] = df['LivestockHoldings'] / (hhsize + EPS)

    if all(c in df.columns for c in ['offfarm_income_USD_PPP_pHH_Yr', 'total_income_USD_PPP_pHH_Yr']):
        offfarm_raw = np.expm1(pd.to_numeric(df['offfarm_income_USD_PPP_pHH_Yr'], errors='coerce').clip(lower=0))
        total_raw   = np.expm1(pd.to_numeric(df['total_income_USD_PPP_pHH_Yr'],   errors='coerce').clip(lower=EPS))
        df['income_diversification'] = (offfarm_raw / total_raw).clip(0, 1)

    if all(c in df.columns for c in ['score_HDDS_GoodSeason', 'score_HDDS_BadSeason']):
        df['hdds_seasonal_gap']     = df['score_HDDS_GoodSeason'] - df['score_HDDS_BadSeason']
        df['hdds_bad_season_share'] = df['score_HDDS_BadSeason'] / (df['score_HDDS_GoodSeason'] + EPS)
    if all(c in df.columns for c in ['score_HDDS_farmbasedGoodSeason', 'score_HDDS_GoodSeason']):
        df['hdds_farm_reliance']    = df['score_HDDS_farmbasedGoodSeason'] / (df['score_HDDS_GoodSeason'] + EPS)
    if all(c in df.columns for c in ['score_HDDS_purchasedGoodSeason', 'score_HDDS_GoodSeason']):
        df['hdds_purchase_reliance'] = df['score_HDDS_purchasedGoodSeason'] / (df['score_HDDS_GoodSeason'] + EPS)

    # Section 4.6 — Interactions, indices, efficiency ratios
    if 'income_per_MAE' in df.columns:
        df['income_per_MAE_sq'] = df['income_per_MAE'] ** 2
    if all(c in df.columns for c in ['LandCultivated', 'LivestockHoldings']):
        df['land_livestock_product'] = df['LandCultivated'] * df['LivestockHoldings']
    if all(c in df.columns for c in ['Market_Orientation', 'farm_income_USD_PPP_pHH_Yr']):
        df['market_income_interaction'] = (
            pd.to_numeric(df['Market_Orientation'], errors='coerce').fillna(0) *
            df['farm_income_USD_PPP_pHH_Yr']
        )
    if all(c in df.columns for c in ['score_HDDS_GoodSeason', 'Food_Self_Sufficiency_kCal_MAE_day']):
        df['diet_quality_quantity'] = df['score_HDDS_GoodSeason'] * df['Food_Self_Sufficiency_kCal_MAE_day']
    if all(c in df.columns for c in ['NrofMonthsWildFoodCons', 'score_HDDS_BadSeason']):
        df['wild_food_coping'] = (
            pd.to_numeric(df['NrofMonthsWildFoodCons'], errors='coerce').fillna(0) *
            (1 / (df['score_HDDS_BadSeason'] + 1))
        )
    if all(c in df.columns for c in ['PPI_Likelihood', 'total_income_USD_PPP_pHH_Yr']):
        df['poverty_income_ratio'] = df['PPI_Likelihood'] / (df['total_income_USD_PPP_pHH_Yr'] + EPS)

    # Aggregated indices
    liv_cols = [c for c in ['LandCultivated', 'LivestockHoldings', 'total_income_USD_PPP_pHH_Yr']
                if c in df.columns]
    if len(liv_cols) >= 2:
        tmp = df[liv_cols].copy().apply(pd.to_numeric, errors='coerce')
        for c in liv_cols:
            r = tmp[c].max() - tmp[c].min()
            tmp[c] = (tmp[c] - tmp[c].min()) / r if r > 0 else 0
        df['livelihood_index'] = tmp.mean(axis=1)

    if all(c in df.columns for c in ['score_HDDS_GoodSeason', 'score_HDDS_BadSeason']):
        df['nutrition_index'] = (df['score_HDDS_GoodSeason'] + df['score_HDDS_BadSeason']) / 2

    env_cols = [c for c in ['GHGEmissions', 'NFertInput'] if c in df.columns]
    if len(env_cols) == 2:
        tmp = df[env_cols].copy().apply(pd.to_numeric, errors='coerce')
        for c in env_cols:
            r = tmp[c].max() - tmp[c].min()
            tmp[c] = (tmp[c] - tmp[c].min()) / r if r > 0 else 0
        df['environmental_index'] = tmp.mean(axis=1)

    if all(c in df.columns for c in ['farm_income_USD_PPP_pHH_Yr', 'LandCultivated']):
        df['farm_income_per_land'] = df['farm_income_USD_PPP_pHH_Yr'] / (df['LandCultivated'] + EPS)
    if all(c in df.columns for c in ['livestock_prodsales_USD_PPP_pHH_Yr', 'LivestockHoldings']):
        df['livestock_productivity'] = df['livestock_prodsales_USD_PPP_pHH_Yr'] / (df['LivestockHoldings'] + EPS)
    if all(c in df.columns for c in ['offfarm_income_USD_PPP_pHH_Yr', 'farm_income_USD_PPP_pHH_Yr']):
        df['offfarm_to_farm_ratio'] = df['offfarm_income_USD_PPP_pHH_Yr'] / (df['farm_income_USD_PPP_pHH_Yr'] + EPS)
    if all(c in df.columns for c in ['LandCultivated', 'LandOwned']):
        df['land_utilization_rate'] = (df['LandCultivated'] / (df['LandOwned'] + EPS)).clip(0, 2)

    return df


# ==============================================================================
# PREPROCESSING & PREDICTION
# ==============================================================================

def build_raw_frame(user_responses: dict) -> pd.DataFrame:
    all_base = NUMERICAL_FEATURES + list(CATEGORICAL_FEATURES.keys())
    data = {col: np.nan for col in all_base}
    data.update(user_responses)
    return pd.DataFrame([data])


def _detect_numeric_pipeline_cols(preprocessor) -> set:
    """Return the set of column names routed to a numeric (median-imputing) sub-pipeline."""
    numeric_cols = set()
    try:
        for _name, transformer, cols in preprocessor.transformers_:
            is_numeric_pipe = False
            if hasattr(transformer, 'steps'):
                for _sname, step in transformer.steps:
                    if hasattr(step, 'strategy') and getattr(step, 'strategy', '') == 'median':
                        is_numeric_pipe = True
                        break
            elif hasattr(transformer, 'strategy') and getattr(transformer, 'strategy', '') == 'median':
                is_numeric_pipe = True
            if is_numeric_pipe and isinstance(cols, (list, np.ndarray)):
                numeric_cols.update(cols)
    except Exception:
        pass
    return numeric_cols


def prepare_sklearn_input(engineered_df: pd.DataFrame, preprocessor):
    df = engineered_df.copy()

    expected_cols = getattr(preprocessor, 'feature_names_in_', None)
    if expected_cols is not None:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(expected_cols)]

    # If categorical columns appear in the numeric sub-pipeline it means the
    # notebook ordinal-encoded them before fitting the preprocessor.
    # Encode them to integer indices using the known category lists so the
    # median imputer receives numeric data as it did during training.
    numeric_pipe_cols = _detect_numeric_pipeline_cols(preprocessor)
    for col, cats in CATEGORICAL_FEATURES.items():
        if col in numeric_pipe_cols and col in df.columns:
            df[col] = df[col].apply(
                lambda v: float(cats.index(v)) if v in cats else np.nan
            )

    return preprocessor.transform(df)


def prepare_catboost_input(engineered_df: pd.DataFrame, num_imputer, feature_meta: dict):
    cb_cols   = feature_meta.get('feature_cols_cb', [])
    num_feats = feature_meta.get('num_features', [])
    cat_feats = feature_meta.get('cat_features', [])

    for col in cb_cols:
        if col not in engineered_df.columns:
            engineered_df[col] = np.nan

    num_present = [c for c in num_feats if c in engineered_df.columns]
    if num_present:
        num_input = engineered_df[num_present].copy()
        # Ordinal-encode any categorical column that ended up in the numeric
        # imputer (it was label-encoded before the imputer was fitted in the
        # notebook, so it expects integers, not raw strings).
        for col, cats in CATEGORICAL_FEATURES.items():
            if col in num_input.columns:
                num_input[col] = num_input[col].apply(
                    lambda v: float(cats.index(v)) if v in cats else np.nan
                )
        num_arr = num_imputer.transform(num_input)
        num_df  = pd.DataFrame(num_arr, columns=num_present)
    else:
        num_df = pd.DataFrame()

    cat_present = [c for c in cat_feats if c in engineered_df.columns]
    cat_df = engineered_df[cat_present].fillna('missing').astype(str).reset_index(drop=True)
    num_df = num_df.reset_index(drop=True)

    combined   = pd.concat([num_df, cat_df], axis=1)
    final_cols = [c for c in cb_cols if c in combined.columns]
    return combined[final_cols]


def run_prediction(model_name, user_responses,
                   models, preprocessor, num_imputer, label_encoder, feature_meta):
    try:
        model = models.get(model_name)
        if model is None:
            return None, None, f"Model '{model_name}' not loaded. Run notebook section 12 to export savedModels/."

        requires_cb = MODEL_INFO[model_name]['requires_catboost_path']
        raw_df      = build_raw_frame(user_responses)
        engineered  = engineer_features(raw_df)

        if requires_cb:
            if num_imputer is None or feature_meta is None:
                return None, None, "CatBoost preprocessing artifacts (num_imputer / feature_metadata) not loaded."
            X = prepare_catboost_input(engineered, num_imputer, feature_meta)
        else:
            if preprocessor is None:
                return None, None, "Sklearn preprocessor not loaded."
            X = prepare_sklearn_input(engineered, preprocessor)

        pred_encoded = model.predict(X)
        if hasattr(pred_encoded, 'flatten'):
            pred_encoded = pred_encoded.flatten()
        pred_int   = int(pred_encoded[0])
        probas_raw = model.predict_proba(X)[0]

        pred_class  = label_encoder.inverse_transform([pred_int])[0]
        probas_dict = {cls: float(p) for cls, p in zip(label_encoder.classes_, probas_raw)}

        return pred_class, probas_dict, None

    except Exception as e:
        return None, None, f"{str(e)}\n\n{traceback.format_exc()}"


# ==============================================================================
# SHAP EXPLAINABILITY
# ==============================================================================

def compute_shap_local(model_name, user_responses,
                       models, preprocessor, num_imputer, label_encoder, feature_meta):
    try:
        import shap as shap_lib
    except ImportError:
        return None, "shap package not installed. Run: pip install shap"

    try:
        model = models.get(model_name)
        if model is None:
            return None, "Model not loaded."

        requires_cb = MODEL_INFO[model_name]['requires_catboost_path']
        raw_df      = build_raw_frame(user_responses)
        engineered  = engineer_features(raw_df)

        if requires_cb:
            X            = prepare_catboost_input(engineered, num_imputer, feature_meta)
            explainer    = shap_lib.TreeExplainer(model)
            shap_values  = explainer.shap_values(X)
            feature_names = X.columns.tolist()
        else:
            X_proc        = prepare_sklearn_input(engineered, preprocessor)
            explainer     = shap_lib.TreeExplainer(model)
            shap_values   = explainer.shap_values(X_proc)
            feature_names = (
                preprocessor.get_feature_names_out().tolist()
                if hasattr(preprocessor, 'get_feature_names_out')
                else [f"f{i}" for i in range(X_proc.shape[1])]
            )

        sv = np.array(shap_values)
        if sv.ndim == 3:
            mean_abs = np.abs(sv[0]).mean(axis=-1)
        elif sv.ndim == 2:
            mean_abs = np.abs(sv[0])
        else:
            mean_abs = np.abs(sv).flatten()[:len(feature_names)]

        shap_df = pd.DataFrame({
            'feature':  feature_names[:len(mean_abs)],
            'abs_shap': mean_abs[:len(feature_names)],
        }).sort_values('abs_shap', ascending=False).reset_index(drop=True)

        return shap_df, None

    except Exception as e:
        return None, f"SHAP computation failed: {str(e)}"


# ==============================================================================
# SESSION STATE
# ==============================================================================

def initialize_session_state():
    defaults = {
        "messages":                [],
        "selected_model":          "Stacking Ensemble",
        "model_selected":          False,
        "current_feature_index":   0,
        "user_responses":          {},
        "all_features_collected":  False,
        "prediction_made":         False,
        "prediction_result":       None,
        "show_shap":               False,
        "prev_question_timestamp": datetime.datetime.fromtimestamp(0),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==============================================================================
# FEATURE COLLECTION HELPERS
# ==============================================================================

def get_all_features_in_order():
    features = []
    for block in FEATURE_BLOCKS.values():
        features.extend(block)
    return features


def get_current_feature():
    all_f = get_all_features_in_order()
    idx   = st.session_state.current_feature_index
    return all_f[idx] if idx < len(all_f) else None


def get_progress_info():
    return st.session_state.current_feature_index, len(get_all_features_in_order())


def validate_numerical_input(feature_name, value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None, "Please enter a valid number."

    bounded = {
        'PPI_Likelihood': (0, 100),
        'Gender_FemaleControl': (0, 1),
        'Market_Orientation': (0, 1),
        'score_HDDS_GoodSeason': (0, 12),
        'score_HDDS_BadSeason': (0, 12),
        'score_HDDS_farmbasedGoodSeason': (0, 12),
        'score_HDDS_purchasedGoodSeason': (0, 12),
        'score_HDDS_farmbasedBadSeason': (0, 12),
        'score_HDDS_purchasedBadSeason': (0, 12),
        'NrofMonthsWildFoodCons': (0, 12),
    }
    if feature_name in bounded:
        lo, hi = bounded[feature_name]
        if not (lo <= v <= hi):
            return None, f"Value must be between {lo} and {hi}."
    elif v < 0:
        return None, "Value cannot be negative."
    return v, None


def validate_categorical_input(feature_name, value):
    opts = CATEGORICAL_FEATURES.get(feature_name, [])
    if value not in opts:
        return None, "Please select a valid option."
    return value, None


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def display_initial_screen(models):
    st.markdown("## 🌾 Food Security Prediction Assistant")
    st.markdown("Predict household food security status using machine learning.")
    st.markdown("---")
    st.markdown("### 🤖 Select a Model")

    available   = [n for n, m in models.items() if m is not None]
    unavailable = [n for n, m in models.items() if m is None]

    if unavailable:
        st.info(
            f"ℹ️ The following models are not yet available "
            f"(run notebook section 12 to export to savedModels/): "
            f"{', '.join(unavailable)}"
        )

    pill_options = [f"{MODEL_INFO[n]['icon']} {n}" for n in MODEL_INFO]
    selected_pill = st.pills(
        label="Available Models",
        label_visibility="collapsed",
        options=pill_options,
    )

    if selected_pill:
        model_name = None
        for name in MODEL_INFO:
            if name in selected_pill:
                model_name = name
                break
        if model_name is None:
            model_name = list(MODEL_INFO.keys())[0]

        if models.get(model_name) is None:
            st.error(f"❌ {model_name} is not loaded yet. Please choose an available model.")
        else:
            info = MODEL_INFO[model_name]
            st.session_state.selected_model  = model_name
            st.session_state.model_selected  = True
            st.session_state.messages.append({"role": "user",      "content": f"I want to use {model_name}"})
            st.session_state.messages.append({"role": "assistant", "content": (
                f"Great choice! I'll use **{model_name}** — {info['description']}.\n\n"
                f"**Weighted Kappa:** {info['weighted_kappa']:.4f} &nbsp;|&nbsp; "
                f"**ROC-AUC:** {info['roc_auc']:.4f}\n\n"
                f"I'll ask you about your household across 8 topic areas. "
                f"Questions marked **(Optional)** can be skipped — those fields will be "
                f"estimated from patterns in the training data. Ready?"
            )})
            st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("📋 Legal Disclaimer", key="disclaimer_btn"):
            show_disclaimer_dialog()


@st.dialog("Legal Disclaimer")
def show_disclaimer_dialog():
    st.markdown("""
### ⚠️ Important Information

**Data Collection:**
- We collect household survey features aligned with the RHoMIS indicator set
- Optional features are imputed from statistical patterns in training data
- Leakage variables (e.g. *NrofMonthsFoodInsecure*) are intentionally excluded

**Data Privacy & Security:**
- Household information is used solely for this prediction session
- Do not enter precise GPS coordinates or personally identifiable information
- Data is processed locally and not stored after the session ends

**Model Limitations:**
- Predictions are probabilistic estimates based on ~13,000 RHoMIS households (2015–2018)
- Primary metric is Quadratic Weighted Kappa (ordinal accuracy), not binary classification
- MildlyFI (~18% recall) is the hardest class to distinguish from adjacent categories
- Performance varies by country — validate locally before operational use

**Disclaimer:**
- Predictions may be inaccurate or biased, especially outside the training distribution
- Not intended as the sole basis for resource allocation or policy decisions

For more information: [RHoMIS Documentation](https://www.rhomis.org)
    """)
    if st.button("I Understand & Accept", key="accept_disclaimer"):
        st.toast("Disclaimer accepted!", icon="✅")


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.container()
            st.markdown(message["content"])


def display_result_panel(pred_class: str, probas_dict: dict, model_name: str):
    meta  = CLASS_META[pred_class]
    info  = MODEL_INFO[model_name]
    conf  = probas_dict.get(pred_class, 0) * 100
    color = meta['color']

    st.markdown(f"""
---
### {meta['emoji']} Assessment Complete

<div style="background:{color}22; border-left:5px solid {color}; padding:14px 18px; border-radius:6px; margin-bottom:12px;">
<h3 style="margin:0; color:{color};">{meta['emoji']} {meta['label']}</h3>
<p style="margin:4px 0 0 0; color:#444;">{meta['desc']}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        f"**Confidence:** {conf:.1f}% &nbsp;|&nbsp; "
        f"**Model:** {model_name} &nbsp;|&nbsp; "
        f"**W-Kappa:** {info['weighted_kappa']:.4f} &nbsp;|&nbsp; "
        f"**ROC-AUC:** {info['roc_auc']:.4f}"
    )

    st.markdown("#### Class Probabilities")
    for cls in HFIAS_ORDER:
        p     = probas_dict.get(cls, 0)
        cmeta = CLASS_META[cls]
        st.markdown(
            f"**{cmeta['label']}** &nbsp; {p*100:.1f}%  "
            f"<div style='background:#eee; border-radius:4px; height:12px; margin-bottom:6px;'>"
            f"<div style='background:{cmeta['color']}; width:{p*100:.1f}%; height:12px; border-radius:4px;'></div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
### 📝 Interpretation

This uses a **4-class ordinal model** evaluated with **Quadratic Weighted Kappa**, which penalises
misclassifications proportionally to their ordinal distance.

**Key drivers (SHAP analysis of top features):**
- `score_HDDS_BadSeason` — lean-season dietary diversity is the single most predictive feature
- `hdds_seasonal_gap` — large good→lean diet drop signals seasonal vulnerability
- `Food_Self_Sufficiency_kCal_MAE_day` — threshold effect below subsistence level
- `PPI_Likelihood`, income features — important but secondary to nutrition indicators

### ✨ Recommended Actions

| Class | Action |
|---|---|
| **Food Secure** | Maintain resilience; focus on lean-season buffers |
| **Mildly FI** | Dietary diversification; off-farm income support |
| **Moderately FI** | Food access programmes; livelihood diversification |
| **Severely FI** | Immediate food assistance; safety-net enrolment |

---
*This prediction is for informational purposes only and should not be the sole basis for critical decisions.*
""")


# ==============================================================================
# MAIN APP FLOW
# ==============================================================================

def main():
    initialize_session_state()
    models, preprocessor, num_imputer, label_encoder, feature_meta = load_artifacts()

    if not st.session_state.model_selected:
        display_initial_screen(models)
        st.stop()

    # Header
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        info = MODEL_INFO[st.session_state.selected_model]
        st.markdown(
            f"### {info['icon']} {st.session_state.selected_model} "
            f"<span style='font-size:0.75em; color:gray;'>"
            f"W-Kappa {info['weighted_kappa']:.4f} · ROC-AUC {info['roc_auc']:.4f}"
            f"</span>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("↻ Restart", use_container_width=True):
            for k in ["messages", "user_responses", "current_feature_index",
                      "model_selected", "all_features_collected",
                      "prediction_made", "prediction_result", "show_shap"]:
                if k == "messages":
                    st.session_state[k] = []
                elif k == "user_responses":
                    st.session_state[k] = {}
                elif k == "current_feature_index":
                    st.session_state[k] = 0
                elif k in ["prediction_result"]:
                    st.session_state[k] = None
                else:
                    st.session_state[k] = False
            st.session_state.selected_model = "Stacking Ensemble"
            st.rerun()

    st.markdown("---")
    display_chat_history()

    # Feature collection
    if not st.session_state.all_features_collected:
        current_feature = get_current_feature()

        if current_feature is None:
            st.session_state.all_features_collected = True
            st.rerun()

        question    = FEATURE_QUESTIONS.get(current_feature, f"Enter {current_feature.replace('_', ' ')}:")
        description = FEATURE_DESCRIPTIONS.get(current_feature, "")
        is_optional = current_feature in OPTIONAL_FEATURES

        current_idx, total = get_progress_info()
        st.progress(current_idx / total, text=f"Question {current_idx + 1} of {total}")

        block_label = next(
            (bl for bl, feats in FEATURE_BLOCKS.items() if current_feature in feats), ""
        )
        if block_label:
            st.caption(f"Section: **{block_label}**")

        st.markdown(f"### {question}")
        if description:
            st.caption(description)

        if current_feature in CATEGORICAL_FEATURES:
            user_input = st.selectbox(
                label="Select option:",
                options=CATEGORICAL_FEATURES[current_feature],
                label_visibility="collapsed",
                key=f"input_{current_feature}",
            )
        else:
            user_input = st.number_input(
                label="Enter value:",
                value=0.0,
                step=0.1,
                label_visibility="collapsed",
                key=f"input_{current_feature}",
            )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            submit_btn = st.button("✓ Submit", use_container_width=True, key=f"submit_{current_feature}")
        with col_c:
            skip_btn = st.button("⏭ Skip", use_container_width=True, key=f"skip_{current_feature}") if is_optional else False

        if skip_btn:
            st.session_state.messages.append({
                "role": "user",
                "content": f"*{question}*\n**Skipped** (will be estimated from data)"
            })
            st.session_state.current_feature_index += 1
            next_feat = get_current_feature()
            ack = f"↩️ Skipped — I'll estimate that. ({current_idx + 1}/{total})"
            if next_feat is None:
                st.session_state.all_features_collected = True
                ack += "\n\n🎉 All done! Running the prediction pipeline..."
            else:
                ack += "\n\nReady for the next question!"
            st.session_state.messages.append({"role": "assistant", "content": ack})
            st.rerun()

        if submit_btn:
            now  = datetime.datetime.now()
            diff = now - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = now
            if diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep((MIN_TIME_BETWEEN_REQUESTS - diff).total_seconds())

            if current_feature in CATEGORICAL_FEATURES:
                validated, err = validate_categorical_input(current_feature, user_input)
            else:
                validated, err = validate_numerical_input(current_feature, user_input)

            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"*{question}*\n**Answer:** {user_input}"
                })
                st.session_state.user_responses[current_feature] = validated
                st.session_state.current_feature_index += 1
                next_feat = get_current_feature()

                ack = f"✅ Got it! ({current_idx + 1}/{total})"
                if next_feat is None:
                    st.session_state.all_features_collected = True
                    ack += "\n\n🎉 All information collected. Running the prediction pipeline..."
                else:
                    ack += "\n\nReady for the next question!"

                st.session_state.messages.append({"role": "assistant", "content": ack})
                st.rerun()

    # Prediction
    elif not st.session_state.prediction_made:
        with st.chat_message("assistant"):
            with st.spinner("🔄 Engineering features and running prediction..."):
                pred_class, probas_dict, err = run_prediction(
                    st.session_state.selected_model,
                    st.session_state.user_responses,
                    models, preprocessor, num_imputer, label_encoder, feature_meta,
                )

            if err:
                msg = f"❌ Prediction failed:\n```\n{err}\n```"
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                st.session_state.prediction_result = (pred_class, probas_dict)
                display_result_panel(pred_class, probas_dict, st.session_state.selected_model)
                cmeta = CLASS_META[pred_class]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"**Prediction:** {cmeta['emoji']} {cmeta['label']} "
                        f"({probas_dict.get(pred_class, 0)*100:.1f}% confidence)\n\n"
                        f"See the panel above for full class probabilities, model metrics, "
                        f"and interpretation. Use the **Explain** button below for SHAP feature contributions."
                    )
                })

        st.session_state.prediction_made = True
        st.rerun()

    # Post-prediction
    else:
        if st.session_state.prediction_result:
            pred_class, probas_dict = st.session_state.prediction_result
            display_result_panel(pred_class, probas_dict, st.session_state.selected_model)

        st.markdown("---")

        # SHAP toggle
        shap_label = "🔍 Hide SHAP explanation" if st.session_state.show_shap else "🔍 Explain this prediction (SHAP)"
        if st.button(shap_label, key="shap_btn"):
            st.session_state.show_shap = not st.session_state.show_shap
            st.rerun()

        if st.session_state.show_shap:
            with st.spinner("Computing SHAP values..."):
                shap_df, shap_err = compute_shap_local(
                    st.session_state.selected_model,
                    st.session_state.user_responses,
                    models, preprocessor, num_imputer, label_encoder, feature_meta,
                )
            if shap_err:
                st.warning(f"⚠️ SHAP unavailable: {shap_err}")
            else:
                st.markdown("#### 🔍 Top Feature Contributions (mean |SHAP| across all classes)")
                top_n     = min(15, len(shap_df))
                max_val   = shap_df['abs_shap'].max()
                for _, row in shap_df.head(top_n).iterrows():
                    bar_pct = (row['abs_shap'] / max_val * 100) if max_val > 0 else 0
                    st.markdown(
                        f"**{row['feature']}** &nbsp; `{row['abs_shap']:.4f}`  "
                        f"<div style='background:#e3f2fd; border-radius:4px; height:10px; margin-bottom:5px;'>"
                        f"<div style='background:#1565c0; width:{bar_pct:.1f}%; height:10px; border-radius:4px;'></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.caption(
                    "Each bar shows the average absolute SHAP contribution of that feature "
                    "to the model's output across all four food-security classes. "
                    "Longer bar = more influential for this household's prediction."
                )

        # Follow-up
        st.markdown("---")
        user_input = st.chat_input(
            placeholder="Ask a follow-up question about your result...",
            key="followup_input"
        )
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                response = (
                    "For on-the-ground guidance, please contact your local agricultural extension service. "
                    "For methodology questions, refer to the [RHoMIS documentation](https://www.rhomis.org) "
                    "and the study notebook."
                )
                st.markdown(response)
            st.session_state.messages.append({"role": "user",      "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
