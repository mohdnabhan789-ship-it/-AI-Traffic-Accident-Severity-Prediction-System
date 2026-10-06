import streamlit as st
import requests


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Traffic Accident Risk Prediction",
    page_icon="🚦",
    layout="wide"
)


# ---------------------------------------------------------
# API configuration
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🚦 AI Traffic Accident Risk Prediction")
st.markdown(
    "Machine Learning based accident severity prediction "
    "with SHAP explainability."
)

st.divider()


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.subheader("Enter Accident Conditions")

col1, col2 = st.columns(2)


with col1:

    police_force = st.number_input(
        "Police Force",
        min_value=1,
        value=1,
        step=1
    )

    speed_limit = st.number_input(
        "Speed Limit",
        min_value=10,
        max_value=120,
        value=60,
        step=10
    )

    light_conditions = st.selectbox(
        "Light Conditions",
        [
            "Daylight: Street light present",
            "Daylight: Street light absent",
            "Darkness: Street lights present and lit",
            "Darkness: Street lights present but unlit",
            "Darkness: No street lighting"
        ]
    )

    weather_conditions = st.selectbox(
        "Weather Conditions",
        [
            "Fine without high winds",
            "Raining without high winds",
            "Raining with high winds",
            "Snowing without high winds",
            "Snowing with high winds",
            "Fog or mist",
            "Other"
        ]
    )


with col2:

    road_surface = st.selectbox(
        "Road Surface Conditions",
        [
            "Dry",
            "Wet/Damp",
            "Frost/Ice",
            "Snow",
            "Flood over 3cm. deep"
        ]
    )

    special_conditions = st.selectbox(
        "Special Conditions at Site",
        [
            "None",
            "Roadworks",
            "Road surface defective",
            "Oil or diesel",
            "Mud",
            "Involvement with ditch"
        ]
    )

    carriageway_hazards = st.selectbox(
        "Carriageway Hazards",
        [
            "None",
            "Other object in carriageway",
            "Any animal in carriageway",
            "Involvement with previous accident",
            "Pedestrian in carriageway",
            "Dislodged vehicle load"
        ]
    )

    urban_rural = st.selectbox(
        "Area Type",
        [
            "Urban",
            "Rural"
        ]
    )

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2030,
        value=2020,
        step=1
    )


# ---------------------------------------------------------
# Convert urban/rural value
# ---------------------------------------------------------

urban_rural_value = 1 if urban_rural == "Urban" else 2


# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict Accident Severity",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if predict_button:

    input_data = {
        "Police_Force": police_force,
        "Speed_limit": speed_limit,
        "Light_Conditions": light_conditions,
        "Weather_Conditions": weather_conditions,
        "Road_Surface_Conditions": road_surface,
        "Special_Conditions_at_Site": special_conditions,
        "Carriageway_Hazards": carriageway_hazards,
        "Urban_or_Rural_Area": urban_rural_value,
        "Year": year
    }

    try:

        # ---------------------------------------------
        # Call prediction API
        # ---------------------------------------------

        prediction_response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            timeout=30
        )

        prediction_response.raise_for_status()

        prediction_data = prediction_response.json()

        severity = prediction_data[
            "predicted_accident_severity"
        ]


        # ---------------------------------------------
        # Call SHAP API
        # ---------------------------------------------

        explain_response = requests.post(
            f"{API_URL}/explain",
            json=input_data,
            timeout=30
        )

        explain_response.raise_for_status()

        explain_data = explain_response.json()

        top_factors = explain_data.get(
            "top_factors",
            []
        )


        # ---------------------------------------------
        # Display prediction
        # ---------------------------------------------

        st.divider()

        st.subheader("Prediction Result")


        if severity == 1:

            st.error(
                "🔴 Fatal Accident Severity"
            )

        elif severity == 2:

            st.warning(
                "🟠 Serious Accident Severity"
            )

        else:

            st.success(
                "🟢 Slight Accident Severity"
            )


        st.metric(
            "Predicted Accident Severity",
            severity
        )


        # ---------------------------------------------
        # SHAP explanation
        # ---------------------------------------------

        st.subheader(
            "Why did the model make this prediction?"
        )

        if top_factors:

            for factor in top_factors:

                feature_name = factor["feature"]
                impact = factor["impact"]
                direction = factor["direction"]

                # Make technical SHAP names easier to read
                readable_name = feature_name

                readable_name = readable_name.replace(
                    "categorical__",
                    ""
                )

                readable_name = readable_name.replace(
                    "numeric__",
                    ""
                )

                readable_name = readable_name.replace(
                    "_",
                    " "
                )

                st.write(
                    f"**{readable_name.title()}**"
                )

                st.caption(
                    f"Impact: {impact} | "
                    f"Direction: {direction}"
                )

                st.progress(
                    min(abs(float(impact)), 1.0)
                )

        else:

            st.info(
                "No explanation factors available."
            )


        # ---------------------------------------------
        # Prediction saved message
        # ---------------------------------------------

        if prediction_data.get(
            "prediction_saved"
        ):

            st.success(
                "Prediction saved successfully to SQLite."
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )


    except requests.exceptions.RequestException as error:

        st.error(
            f"API Error: {error}"
        )


    except Exception as error:

        st.error(
            f"Unexpected error: {error}"
        )