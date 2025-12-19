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
Food Security Prediction Assistant - Chat-Based ML Deployment App

A Streamlit application for deploying machine learning models with a conversational
chat interface. Users can select a model and provide household information to receive
food security predictions with confidence scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import time
from pathlib import Path
from collections import namedtuple

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Food Security Prediction Assistant",
    page_icon="🌾",
    # layout="",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

# Model configuration
MODEL_INFO = {
    "Random Forest": {
        "file": "deployment/optimizedModels/random_forest_optimized.pkl",
        "icon": ":blue[:material/hub:]",
        "description": "Fast and reliable ensemble method",
        "accuracy": 0.80
    },
    "XGBoost": {
        "file": "deployment/optimizedModels/xgboost_optimized.pkl",
        "icon": ":green[:material/leaf:]",
        "description": "High accuracy gradient boosting",
        "accuracy": 0.82
    },
    "LightGBM": {
        "file": "deployment/optimizedModels/lightgbm_optimized.pkl",
        "icon": ":orange[:material/trending_up:]",
        "description": "Lightweight and fast boosting",
        "accuracy": 0.81
    },
    "Logistic Regression": {
        "file": "deployment/optimizedModels/logistic_regression_optimized.pkl",
        "icon": "⚙️",
        "description": "Interpretable baseline model",
        "accuracy": 0.75
    }
}

# Feature configuration - Top 12 Core Features (based on feature importance analysis)
NUMERICAL_FEATURES = [
    'NrofMonthsFoodInsecure',           # Food security status (RF, XGBoost, LightGBM)
    'PPI_Likelihood',                   # Poverty assessment (RF, LightGBM)
    'Food_Self_Sufficiency_kCal_MAE_day', # Food sufficiency (RF, LightGBM)
    'HHsizeMAE',                        # Household size (RF, LightGBM)
    'LivestockHoldings',                # Livestock holdings (RF, LightGBM)
    'score_HDDS_GoodSeason',            # Dietary diversity (RF, LightGBM)
    'farm_income_USD_PPP_pHH_Yr',       # Farm income (RF, XGBoost, LogReg)
    'LandOwned',                        # Land resources (RF, LightGBM)
    'TVA_USD_PPP_pmae_pday',            # Total value of activities (RF, LightGBM)
    'Gender_FemaleControl'              # Gender considerations (LightGBM)
]

CATEGORICAL_FEATURES = {
    'Country': ['Tanzania', 'Burkina_Faso', 'Cambodia', 'DRC', 'Ethiopia',
                'Ghana', 'India', 'Kenya', 'LaoPDR', 'Malawi', 'Mali',
                'Nicaragua', 'Peru', 'Uganda', 'Vietnam', 'Zambia',
                'Burundi', 'El_Salvador', 'Costa_Rica', 'Guatemala', 'Honduras'],
    'WorstFoodSecMonth': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                          'sep', 'oct', 'nov', 'dec', 'N/A']
}

# Feature descriptions for user guidance
FEATURE_DESCRIPTIONS = {
    'Country': 'Select the country where your household is located',
    'NrofMonthsFoodInsecure': 'Think about the past 12 months - were there times you didn\'t have enough food?',
    'WorstFoodSecMonth': 'Many households have a "hungry season" - when is it for you?',
    'HHsizeMAE': '💡 Tip: Count all family members. Adjust for children (they eat less than adults)',
    'LandOwned': '💡 Tip: Include all land your family owns/controls. 1 hectare is about the size of a small farm plot',
    'LivestockHoldings': '💡 Tip: List what you own. Cattle count more than goats, which count more than chickens',
    'Food_Self_Sufficiency_kCal_MAE_day': '💡 Tip: Consider meals your household produces + buys. A typical day: 3 meals = 1500-2000 calories',
    'score_HDDS_GoodSeason': 'Food groups: grains, legumes, vegetables, fruits, meat, fish, eggs, dairy, oils, sweets, spices, etc.',
    'farm_income_USD_PPP_pHH_Yr': '💡 Tip: Total earnings from crops + livestock sales over the whole year. Rough estimates are fine!',
    'TVA_USD_PPP_pmae_pday': '💡 Tip: Typical daily income from farming or other work. If unsure, estimate based on your usual earnings',
    'PPI_Likelihood': '💡 Tip: Based on your land, assets, income, and education - could you improve your situation?',
    'Gender_FemaleControl': '💡 Tip: Who decides what to grow, how to spend money, children\'s education, etc.?'
}

# Feature grouping for conversational flow
FEATURE_BLOCKS = {
    "Location": ['Country'],
    "Household Information": ['HHsizeMAE'],
    "Land Resources": ['LandOwned'],
    "Livestock": ['LivestockHoldings'],
    "Food Security Status": ['NrofMonthsFoodInsecure', 'WorstFoodSecMonth'],
    "Nutritional Status": ['Food_Self_Sufficiency_kCal_MAE_day', 'score_HDDS_GoodSeason'],
    "Income & Livelihoods": ['farm_income_USD_PPP_pHH_Yr', 'TVA_USD_PPP_pmae_pday'],
    "Poverty & Gender": ['PPI_Likelihood', 'Gender_FemaleControl']
}

# Rate limiting
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

# Debug mode
DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

# Named tuple for task management
TaskInfo = namedtuple("TaskInfo", ["name", "value"])

# ==============================================================================
# MODEL LOADING (PLACEHOLDER - REPLACE WITH ACTUAL MODELS)
# ==============================================================================

@st.cache_resource(ttl=3600)
def load_models():
    """
    Load all ML models, preprocessor, and label encoder.
    Models are loaded from the optimizedModels folder.
    Only display messages if there are errors or warnings.
    """
    models = {}
    
    for model_name, info in MODEL_INFO.items():
        try:
            model_path = Path(info["file"])
            
            if model_path.exists():
                model = joblib.load(model_path)
                models[model_name] = model
                # No success message - only show errors
            else:
                st.warning(f"⚠️ Model file not found: {info['file']}")
                models[model_name] = None
                
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {str(e)}")
            models[model_name] = None
    
    # Load preprocessor
    try:
        preprocessor_path = Path("deployment/optimizedModels/preprocessor.pkl")
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            # No success message - only show errors
        else:
            st.warning("⚠️ Preprocessor file not found")
            preprocessor = None
    except Exception as e:
        st.error(f"❌ Error loading preprocessor: {str(e)}")
        preprocessor = None
    
    # Load label encoder
    try:
        encoder_path = Path("deployment/optimizedModels/label_encoder.pkl")
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
            # No success message - only show errors
        else:
            st.warning("⚠️ Label encoder file not found")
            label_encoder = None
    except Exception as e:
        st.error(f"❌ Error loading label encoder: {str(e)}")
        label_encoder = None
    
    return models, preprocessor, label_encoder


@st.cache_resource
def get_model_list():
    """Get list of available models."""
    return list(MODEL_INFO.keys())


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "model_selected" not in st.session_state:
        st.session_state.model_selected = False
    if "current_block_index" not in st.session_state:
        st.session_state.current_block_index = 0
    if "current_feature_index" not in st.session_state:
        st.session_state.current_feature_index = 0
    if "user_responses" not in st.session_state:
        st.session_state.user_responses = {}
    if "all_features_collected" not in st.session_state:
        st.session_state.all_features_collected = False
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "prev_question_timestamp" not in st.session_state:
        st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)


def get_all_features_in_order():
    """Get all features in a flat list maintaining block order."""
    all_features = []
    for block in FEATURE_BLOCKS.values():
        all_features.extend(block)
    return all_features


def get_current_feature():
    """Get the current feature to ask for."""
    all_features = get_all_features_in_order()
    if st.session_state.current_feature_index < len(all_features):
        return all_features[st.session_state.current_feature_index]
    return None


def get_feature_question(feature_name):
    """Generate a user-friendly question for a feature."""
    questions = {
        'Country': "Let's start. Which country is your household located in?",
        'HHsizeMAE': "How many household members are there? (Or estimate: 1 adult ≈ 1, 1 child ≈ 0.5)",
        'LandOwned': "How much land does your household own? (Estimate in hectares. 1 hectare ≈ 2.5 acres)",
        'LivestockHoldings': "What livestock do you own? (Rough total: 1 cow=1 TLU, 1 goat=0.2 TLU, 10 chickens=0.1 TLU)",
        'NrofMonthsFoodInsecure': "How many months in the past year did your household struggle to have enough food? (0-12)",
        'WorstFoodSecMonth': "During which month is food security typically worst?",
        'Food_Self_Sufficiency_kCal_MAE_day': "Roughly how much food (in calories) can your household produce/access per day? (Typical: 1500-2500)",
        'score_HDDS_GoodSeason': "During the good season, how many different food groups did your household consume? (0-12 groups)",
        'farm_income_USD_PPP_pHH_Yr': "Roughly, what was your household's annual farm income? (Estimate in USD. If unsure: low=$100-500, medium=$500-2000, high=$2000+)",
        'TVA_USD_PPP_pmae_pday': "Roughly, how much does your household earn per day from all activities? (If you earn ~$1-5/day, estimate accordingly)",
        'PPI_Likelihood': "Based on your circumstances, do you think you could move out of poverty? (0-100%, where 0=unlikely, 100=very likely)",
        'Gender_FemaleControl': "What proportion of household decisions are made by women? (0-1 scale: 0=none, 0.5=half, 1=all)"
    }
    return questions.get(feature_name, f"Please provide a value for {feature_name.replace('_', ' ')}:")


def validate_numerical_input(feature_name, value):
    """Validate numerical input with constraints."""
    try:
        num_value = float(value)
        
        # Define constraints for specific features
        if feature_name == 'NrofMonthsFoodInsecure':
            if not (0 <= num_value <= 12):
                return None, "Please enter a value between 0 and 12 months"
        elif feature_name == 'PPI_Likelihood':
            if not (0 <= num_value <= 100):
                return None, "Please enter a value between 0 and 100%"
        elif feature_name == 'Gender_FemaleControl':
            if not (0 <= num_value <= 1):
                return None, "Please enter a value between 0 and 1 (e.g., 0.5 for 50%)"
        elif feature_name in ['HHsizeMAE', 'LandOwned', 'LivestockHoldings']:
            if num_value < 0:
                return None, f"{feature_name} cannot be negative"
        elif feature_name == 'score_HDDS_GoodSeason':
            if not (0 <= num_value <= 12):
                return None, "Dietary diversity score should be between 0 and 12"
        elif feature_name in ['Food_Self_Sufficiency_kCal_MAE_day', 'TVA_USD_PPP_pmae_pday', 'farm_income_USD_PPP_pHH_Yr']:
            if num_value < 0:
                return None, f"{feature_name} cannot be negative"
        
        return num_value, None
        
    except ValueError:
        return None, "Invalid input: Please enter a valid number"


def validate_categorical_input(feature_name, value):
    """Validate categorical input against known values."""
    if feature_name not in CATEGORICAL_FEATURES:
        return None, f"Unknown categorical feature: {feature_name}"
    
    valid_options = CATEGORICAL_FEATURES[feature_name]
    if value not in valid_options:
        options_str = ', '.join(valid_options[:5]) + ('...' if len(valid_options) > 5 else '')
        return None, f"Invalid option. Please select from available options."
    
    return value, None


def preprocess_user_input(user_responses, preprocessor):
    """
    Convert user responses to a DataFrame and apply preprocessing.
    The preprocessor expects raw feature names (before transformation).
    """
    try:
        if preprocessor is None:
            st.error("❌ Preprocessor not available. Cannot make prediction.")
            return None, "Preprocessor not loaded"
        
        # Define ALL features the training data had (in order)
        # These are the original raw features before any transformation
        all_raw_features = [
            # Numerical features
            'HHsizeMAE', 'LandOwned', 'LandCultivated', 'LivestockHoldings',
            'NrofMonthsFoodInsecure', 'PPI_Threshold', 'PPI_Likelihood',
            'score_HDDS_GoodSeason', 'score_HDDS_farmbasedGoodSeason',
            'score_HDDS_purchasedGoodSeason', 'score_HDDS_BadSeason',
            'score_HDDS_farmbasedBadSeason', 'score_HDDS_purchasedBadSeason',
            'TVA_USD_PPP_pmae_pday', 'total_income_USD_PPP_pHH_Yr',
            'offfarm_income_USD_PPP_pHH_Yr', 'farm_income_USD_PPP_pHH_Yr',
            'value_crop_consumed_USD_PPP_pHH_Yr', 'livestock_prodsales_USD_PPP_pHH_Yr',
            'value_livestock_production_USD_PPP_pHH_Yr', 'Food_Self_Sufficiency_kCal_MAE_day',
            'NrofMonthsWildFoodCons', 'Gender_FemaleControl',
            # Categorical features
            'Country', 'HouseholdType', 'Head_EducationLevel', 
            'WorstFoodSecMonth', 'BestFoodSecMonth',
            'Livestock_Orientation', 'Market_Orientation'
        ]
        
        # Create a dictionary with default values (NaN for missing features)
        input_data = {feature: np.nan for feature in all_raw_features}
        
        # Update with user-provided responses
        input_data.update(user_responses)
        
        # Create DataFrame with all features in the correct order
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in the same order as training
        input_df = input_df[all_raw_features]
        
        st.write(f"ℹ️ Input shape: {input_df.shape}")
        st.write(f"ℹ️ Provided features: {list(user_responses.keys())}")
        st.write(f"ℹ️ Missing features (will be imputed): {len(all_raw_features) - len(user_responses)}")
        
        # Apply preprocessor.transform to the input
        # The preprocessor will handle feature transformation and imputation
        preprocessed_data = preprocessor.transform(input_df)
        
        st.write(f"ℹ️ Output shape after preprocessing: {preprocessed_data.shape}")
        
        return input_df, preprocessed_data
        
    except Exception as e:
        error_msg = f"Error preprocessing data: {str(e)}"
        st.error(f"❌ {error_msg}")
        import traceback
        st.error(traceback.format_exc())
        return None, error_msg


def make_prediction(model, preprocessed_data, label_encoder):
    """
    Make prediction using the selected model.
    """
    try:
        if model is None:
            return None, None, "❌ Model not available"
        
        # Make prediction
        prediction = model.predict(preprocessed_data)[0]
        
        # Get prediction probability
        try:
            probabilities = model.predict_proba(preprocessed_data)[0]
            confidence = max(probabilities) * 100
        except (AttributeError, IndexError):
            # Model doesn't have predict_proba (e.g., some SVM models)
            confidence = 90.0
        
        # Decode prediction using label_encoder
        if label_encoder is None:
            pred_label = "Food Secure" if prediction == 1 else "Food Insecure"
        else:
            decoded_label = label_encoder.inverse_transform([prediction])[0]
            # Ensure proper formatting of the label
            if "Secure" in str(decoded_label) and "Food" not in str(decoded_label):
                pred_label = f"Food {decoded_label}"
            else:
                pred_label = str(decoded_label)
        
        return pred_label, confidence, None
        
    except Exception as e:
        return None, None, f"❌ Error making prediction: {str(e)}"


def display_initial_screen():
    """Display the initial model selection screen."""
    # st.markdown("---")
    
    # Decorative element
    st.markdown("## 🌾 Food Security Prediction Assistant")
    st.markdown("Predict household food security status using machine learning")
    
    st.markdown("---")
    st.markdown("### 🤖 Select a Model")
    st.markdown("Choose which model to use for making your prediction:")
    
    # Display model selection as pills
    model_pills = []
    for model_name, info in MODEL_INFO.items():
        label = f"{info['icon']} {model_name}"
        model_pills.append(label)
    
    selected_pill = st.pills(
        label="Available Models",
        label_visibility="collapsed",
        options=model_pills,
    )
    
    if selected_pill:
        # Extract model name from pill text (remove icon)
        for i, pill in enumerate(model_pills):
            if pill == selected_pill:
                model_name = list(MODEL_INFO.keys())[i]
                st.session_state.selected_model = model_name
                st.session_state.model_selected = True
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"I want to use {model_name}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Great choice! I'll use **{model_name}** ({MODEL_INFO[model_name]['description']}). Let's start collecting your household information. I'll ask you questions one by one. Ready?"
                })
                
                st.rerun()
    
    st.markdown("---")
    
    # Legal disclaimer button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("📋 Legal Disclaimer", key="disclaimer_btn"):
            show_disclaimer_dialog()


@st.dialog("Legal Disclaimer")
def show_disclaimer_dialog():
    """Display legal disclaimer in a modal dialog."""
    st.markdown("""
    ### ⚠️ Important Information
    
    **Data Collection:**
    - We collect only the most important features for accurate predictions
    - Any missing features are automatically estimated based on statistical patterns
    - This approach maintains prediction accuracy while reducing data burden
    
    **Data Privacy & Security:**
    - Your household information will be used solely for prediction
    - Do not enter sensitive personal data (names, precise locations, etc.)
    - Data is processed locally and not stored after prediction
    
    **Model Limitations:**
    - Predictions are based on statistical models and may not be 100% accurate
    - Results should be interpreted in context of local conditions
    - This tool provides estimates only, not definitive assessments
    - Food security is influenced by many factors beyond the data collected here
    
    **Disclaimer:**
    - Predictions may be inaccurate, inefficient, or biased
    - Users should exercise reasonable judgment and human oversight
    - We are not liable for any actions or decisions based on these predictions
    - Use this tool responsibly and ethically
    
    **By proceeding, you agree to:**
    - Provide accurate information to the best of your knowledge
    - Use predictions only for informational purposes
    - Not rely solely on this tool for critical decisions
    
    For more information, visit: [RHoMIS Documentation](https://www.rhomis.org)
    """)
    
    if st.button("I Understand & Accept", key="accept_disclaimer"):
        st.toast("Disclaimer accepted!", icon="✅")


def display_chat_history():
    """Display all messages from chat history."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # Fix ghost message bug
            
            st.markdown(message["content"])


def display_feature_input(feature_name):
    """Display input widget for a specific feature."""
    question = get_feature_question(feature_name)
    description = FEATURE_DESCRIPTIONS.get(feature_name, "")
    
    st.markdown(f"**{question}**")
    if description:
        st.caption(f"{description}")
    
    # Determine input type and create widget
    if feature_name in CATEGORICAL_FEATURES:
        options = CATEGORICAL_FEATURES[feature_name]
        user_input = st.selectbox(
            label="Select option:",
            options=options,
            label_visibility="collapsed",
            key=f"input_{feature_name}"
        )
        return user_input, True
    else:
        user_input = st.number_input(
            label="Enter value:",
            value=0.0,
            step=0.1,
            label_visibility="collapsed",
            key=f"input_{feature_name}"
        )
        return user_input, False


def get_progress_info():
    """Get current progress information."""
    all_features = get_all_features_in_order()
    current_index = st.session_state.current_feature_index
    total_features = len(all_features)
    return current_index, total_features


def display_prediction_result(pred_label, confidence, model_name):
    """Display prediction result in chat format."""
    accuracy = MODEL_INFO[model_name]["accuracy"]
    
    # Determine status emoji
    status_emoji = "✅" if pred_label == "Food Secure" else "⚠️"
    status_color = "green" if pred_label == "Food Secure" else "orange"
    
    result_text = f"""
    ### Assessment Complete!
    
    **Predicted Status:** {status_emoji} **{pred_label}**
    
    **Confidence:** {confidence:.1f}%
    
    **Model Used:** {model_name}
    
    **Model Accuracy:** {accuracy*100:.0f}%
    
    ---
    
    ### 📝 Interpretation
    
    This prediction is based on the household characteristics you provided. 
    Food security is influenced by many factors, and this assessment should be 
    considered as one input among many when making decisions.
    
    ### ✨ Next Steps
    
    - **Understand Your Results:** Review the prediction in context of your local situation
    - **Take Action:** Based on the results, consider appropriate interventions
    - **Get Support:** Reach out to local agricultural extension services if needed
    
    ---
    
    **Disclaimer:** This prediction is for informational purposes only and should 
    not be the sole basis for critical decisions.
    """
    
    return result_text


# ==============================================================================
# MAIN APP FLOW
# ==============================================================================

def main():
    """Main application flow."""
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    models, preprocessor, label_encoder = load_models()
    
    # Display header
    # st.html("<h1 style='text-align: center; margin-bottom: 0;'>🌾 Food Security Prediction Assistant</h1>")
    # st.markdown("<p style='text-align: center; color: gray;'>Using ML to predict household food security</p>", 
    #             unsafe_allow_html=True)
    
    # Check if model is selected
    if not st.session_state.model_selected:
        display_initial_screen()
        st.stop()
    
    # Display header with model name and restart button
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"### 🤖 Model: {st.session_state.selected_model}")
    with col2:
        if st.button("↻ Restart", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_model = None
            st.session_state.model_selected = False
            st.session_state.current_feature_index = 0
            st.session_state.user_responses = {}
            st.session_state.all_features_collected = False
            st.session_state.prediction_made = False
            st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    display_chat_history()
    
    # Handle feature collection
    if not st.session_state.all_features_collected:
        # Get current feature
        current_feature = get_current_feature()
        
        if current_feature is None:
            st.session_state.all_features_collected = True
            st.rerun()
        
        # Display question and get input
        question = get_feature_question(current_feature)
        description = FEATURE_DESCRIPTIONS.get(current_feature, "")
        
        st.markdown(f"### {question}")
        if description:
            st.caption(f"{description}")
        
        # Create input widget based on feature type
        if current_feature in CATEGORICAL_FEATURES:
            # Use selectbox for categorical features
            options = CATEGORICAL_FEATURES[current_feature]
            user_input = st.selectbox(
                label=f"Select {current_feature.replace('_', ' ')}:",
                options=options,
                label_visibility="collapsed",
                key=f"input_{current_feature}"
            )
            input_received = True
        else:
            # Use number input for numerical features
            user_input = st.number_input(
                label=f"Enter {current_feature.replace('_', ' ')}:",
                value=0.0,
                step=0.1,
                label_visibility="collapsed",
                key=f"input_{current_feature}"
            )
            input_received = True
        
        # Create columns for Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.button("✓ Submit", use_container_width=True, key=f"submit_{current_feature}")
        
        if submit_button:
            # Rate limiting
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp
            
            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep((MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds())
            
            # Validate input
            if current_feature in CATEGORICAL_FEATURES:
                validated_input, error_msg = validate_categorical_input(current_feature, user_input)
            else:
                validated_input, error_msg = validate_numerical_input(current_feature, user_input)
            
            if error_msg:
                st.error(f"❌ {error_msg}")
            else:
                # Add to chat history
                st.session_state.messages.append({"role": "user", "content": f"{question}\n**Answer:** {user_input}"})
                
                # Store response
                st.session_state.user_responses[current_feature] = validated_input
                
                # Get progress
                current_idx, total = get_progress_info()
                progress_text = f"({current_idx + 1}/{total})"
                
                # Generate assistant response
                acknowledgment = f"✅ Got it! {progress_text}"
                
                # Move to next feature
                st.session_state.current_feature_index += 1
                next_feature = get_current_feature()
                
                if next_feature is None:
                    response_text = f"{acknowledgment}\n\n🎉 Great! I've collected all the information. Let me make the prediction..."
                    st.session_state.all_features_collected = True
                else:
                    response_text = f"{acknowledgment}\n\nReady for the next question!"
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    elif not st.session_state.prediction_made:
        # Make prediction
        with st.chat_message("assistant"):
            with st.spinner("🔄 Processing your data and making prediction..."):
                # Preprocess data
                input_df, preprocessed_data = preprocess_user_input(
                    st.session_state.user_responses,
                    preprocessor
                )
                
                if input_df is None:
                    error_msg = f"❌ {preprocessed_data}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.prediction_made = True
                else:
                    # Make prediction
                    model = models.get(st.session_state.selected_model)
                    if model is None:
                        error_msg = f"❌ Model '{st.session_state.selected_model}' is not available"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        pred_label, confidence, pred_error = make_prediction(
                            model,
                            preprocessed_data,
                            label_encoder
                        )
                        
                        if pred_error:
                            error_msg = f"❌ {pred_error}"
                            st.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            # Display results
                            result_text = display_prediction_result(
                                pred_label,
                                confidence,
                                st.session_state.selected_model
                            )
                            st.markdown(result_text)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result_text
                            })
                    
                    st.session_state.prediction_made = True
        
        st.rerun()
    
    else:
        # Show follow-up options
        user_input = st.chat_input(
            placeholder="Ask a follow-up question...",
            key="followup_input"
        )
        
        if user_input:
            with st.chat_message("user"):
                st.text(user_input)
            
            with st.chat_message("assistant"):
                response = "Thank you for your question! For more detailed analysis or support, please reach out to your local agricultural extension services."
                st.markdown(response)
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()


# ==============================================================================
# APP ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
