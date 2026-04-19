import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessing objects
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
knn_model = joblib.load('knn_model.pkl')

# Load the original cleaned data to get unique values for dropdowns
df_cleaned = pd.read_csv('cleaned_tour_data.csv')

# Define the categorical and numerical columns based on our previous processing
categorical_cols = ['age_range', 'gender', 'destination', 'transport_mode', 'stayed_overnight', 'travel_season']
numerical_cols = ['main_travel_cost_bdt', 'hotel_cost_per_night_bdt', 'food_cost_per_day_bdt', 'local_transport_cost_per_day_bdt', 'number_of_trip_days', 'number_of_travellers', 'total_trip_cost_bdt']

st.title('Tour Satisfaction Predictor')
st.write('Enter details to predict customer satisfaction level.')

# Create input fields for user
input_data = {}

# Categorical inputs
for col in categorical_cols:
    options = df_cleaned[col].unique().tolist()
    input_data[col] = st.selectbox(f'Select {col.replace("_", " ").title()}', options)

# Numerical inputs
for col in numerical_cols:
    # Get min/max from the original data for range guidance
    min_val = float(df_cleaned[col].min())
    max_val = float(df_cleaned[col].max())
    # Use st.number_input for numerical features
    input_data[col] = st.number_input(f'Enter {col.replace("_", " ").title()}',
                                      min_value=min_val, max_value=max_val,
                                      value=np.mean(df_cleaned[col]), format="%.2f")

if st.button('Predict Satisfaction'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess categorical features
    input_encoded = encoder.transform(input_df[categorical_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Preprocess numerical features
    input_scaled = scaler.transform(input_df[numerical_cols])
    input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols)

    # Combine preprocessed features
    # Ensure column order matches X_processed used during training
    # This is a critical step: create a full processed DataFrame from raw input
    # and then ensure it has ALL columns that X_processed had, filling missing with 0s
    # This relies on the global X_processed from training being available or recreating its columns
    # For simplicity here, we'll assume the order and presence of columns from X_processed
    # A more robust solution might involve saving X_processed columns or a full pipeline

    # Create a dummy dataframe with all columns from training X_processed
    # This is a simplification; in a real app, you'd ensure proper column alignment
    # For this example, we will merge based on known column names

    # To make this robust, we need to know all columns that were in X_processed.
    # Let's get the columns from the X_processed variable (if it's still available)
    # For demonstration, we will rebuild the X_processed structure

    # Reconstruct the X_processed structure from training (numerical_cols + encoded_categorical_cols)
    # Get all column names from the encoder for categorical features
    all_encoded_categorical_cols = encoder.get_feature_names_out(categorical_cols)

    # Create an empty DataFrame with all expected columns
    processed_input_df = pd.DataFrame(columns=numerical_cols + all_encoded_categorical_cols.tolist())

    # Fill in the scaled numerical values
    for col in numerical_cols:
        processed_input_df.loc[0, col] = input_scaled_df.loc[0, col]

    # Fill in the one-hot encoded categorical values
    for col in all_encoded_categorical_cols:
        if col in input_encoded_df.columns:
            processed_input_df.loc[0, col] = input_encoded_df.loc[0, col]
        else:
            processed_input_df.loc[0, col] = 0.0 # Fill non-present encoded cols with 0

    # Convert all columns to numeric (they should be after this process)
    processed_input_df = processed_input_df.astype(float)

    prediction = knn_model.predict(processed_input_df)
    st.success(f'Predicted Satisfaction Level: **{prediction[0]}**')
