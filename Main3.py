import requests
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from fuzzywuzzy import process
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from scikeras.wrappers import KerasRegressor



# Calculate the average value for the corresponding months
def calculate_monthly_average(rain_data, months):
    indices = [month_to_index[month] for month in months]
    values = [rain_data[i] for i in indices]
    return np.mean(values)

def combine_monthly_data(df):
    # Combine the monthly and annual data into a list for each row
    df['Rain_Monthly_Data'] = df[
        ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']].values.tolist()

    # Drop the individual monthly and seasonal columns
    df = df.drop(
        columns=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'JF',
                 'MAM', 'JJAS', 'OND'])

    return df


def import_excel_to_dataframe(file_path):
    df = pd.read_excel(file_path)
    return df


def import_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    return df


def get_current_location():
    try:
        # Send request to ipinfo.io API
        response = requests.get('https://ipinfo.io/json')
        data = response.json()

        # Extract latitude and longitude
        loc = data['loc'].split(',')
        latitude = float(loc[0])
        longitude = float(loc[1])

        # Create address from city, region, and country
        city = data.get('city', 'Unknown City')
        region = data.get('region', 'Unknown Region')
        country = data.get('country', 'Unknown Country')
        address = f"{region}"

        return latitude, longitude, address
    except Exception as e:
        print(f"Error retrieving location: {e}")
        return None, None, None


def predict_best_crop(model, label_encoder, input_params):
    """Predict the best crop to plant based on input parameters.

    Parameters:
    model (LGBMClassifier): Trained LightGBM model.
    label_encoder (LabelEncoder): Fitted label encoder.
    input_params (dict): Dictionary of input parameters.

    Returns:
    str: The best crop to plant."""

    # Convert input parameters to DataFrame
    input_df = pd.DataFrame([input_params])

    # Make prediction
    prediction = model.predict(input_df)

    # Decode the prediction
    best_crop = label_encoder.inverse_transform(prediction)[0]

    return best_crop


def save_model_and_encoder(model, encoder, model_path='best_model.pkl', encoder_path='label_encoder.pkl'):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)


def load_model_and_encoder(model_path='best_model.pkl', encoder_path='label_encoder.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder


def map_crops(df, mapping, label_column='Crop'):
    df[label_column] = df[label_column].map(mapping)
    return df

def safe_eval(val):
    if isinstance(val, str):
        return eval(val)
    return val

def main():
    lat, lng, address = get_current_location()
    if lat and lng:
        print(f"Latitude: {lat}, Longitude: {lng}")
        print(f"Address: {address}")
    else:
        print("Could not retrieve location")

    file_path1 = r"C:/Users/Huw Whitworth/Downloads/Crop_recommendation.xlsx"

    df_Recommendation = import_excel_to_dataframe(file_path1)
    print(df_Recommendation)

    file_path = r"C:/Users/Huw Whitworth/Downloads/crop_yield/crop_yield.csv"

    df_Yield = import_csv_to_dataframe(file_path)
    print(df_Yield)

    file_path = r"C:/Users/Huw Whitworth/Downloads/Sub_Division_IMD_2017.csv"

    df_Rain = import_csv_to_dataframe(file_path)
    print(df_Rain)

    classes = df_Recommendation['label'].unique()
    # Print the unique classes
    print(classes)

    classes_Yield = df_Yield['Crop'].unique()
    # Print the unique classes
    print(classes_Yield)

    crop_mapping = {
        'Arecanut': 'coconut',
        'Arhar/Tur': 'pigeonpeas',
        'Castor seed': 'other oilseeds',
        'Coconut ': 'coconut',
        'Cotton(lint)': 'cotton',
        'Dry chillies': 'other',
        'Gram': 'chickpea',
        'Jute': 'jute',
        'Linseed': 'other oilseeds',
        'Maize': 'maize',
        'Mesta': 'jute',
        'Niger seed': 'other oilseeds',
        'Onion': 'other',
        'Other Rabi pulses': 'other',
        'Potato': 'other',
        'Rapeseed &Mustard': 'other oilseeds',
        'Rice': 'rice',
        'Sesamum': 'other oilseeds',
        'Small millets': 'other',
        'Sugarcane': 'other',
        'Sweet potato': 'other',
        'Tapioca': 'other',
        'Tobacco': 'other',
        'Turmeric': 'other',
        'Wheat': 'other',
        'Bajra': 'other',
        'Black pepper': 'other',
        'Cardamom': 'other',
        'Coriander': 'other',
        'Garlic': 'other',
        'Ginger': 'other',
        'Groundnut': 'other oilseeds',
        'Horse-gram': 'other',
        'Jowar': 'other',
        'Ragi': 'other',
        'Cashewnut': 'coconut',
        'Banana': 'banana',
        'Soyabean': 'other oilseeds',
        'Barley': 'other',
        'Khesari': 'other',
        'Masoor': 'lentil',
        'Moong(Green Gram)': 'mungbean',
        'Other Kharif pulses': 'other',
        'Safflower': 'other oilseeds',
        'Sannhamp': 'other',
        'Sunflower': 'other oilseeds',
        'Urad': 'blackgram',
        'Peas & beans (Pulses)': 'other',
        'other oilseeds': 'other oilseeds',
        'Other Cereals': 'other',
        'Cowpea(Lobia)': 'other',
        'Oilseeds total': 'other oilseeds',
        'Guar seed': 'other',
        'Other Summer Pulses': 'other',
        'Moth': 'mothbeans'
    }

    # Apply the crop mapping
    df_Yield = map_crops(df_Yield, crop_mapping)
    print("Mapped DataFrame:")
    print(df_Yield)

    df_Yield = df_Yield[df_Yield['Crop'].isin(classes)]

    print("\nFiltered yeild:")
    print(df_Yield)

    print(df_Yield['Season'].unique())
    print(df_Yield['Crop_Year'].unique())
    print(df_Yield['State'].unique())

    # Define the mapping from seasons to months
    season_to_months = {
        'Kharif': ['June', 'July', 'August', 'September', 'October'],
        'Rabi': ['October', 'November', 'December', 'January', 'February', 'March'],
        'Autumn': ['September', 'October', 'November', 'December'],
        'Summer': ['April', 'May', 'June'],
        'Winter': ['December', 'January', 'February'],
        'Whole Year': ['Annual']
    }

    # Strip any extra spaces around the season names
    df_Yield['Season'] = df_Yield['Season'].str.strip()

    # Add a new column with the corresponding months
    df_Yield['Months'] = df_Yield['Season'].map(season_to_months)

    # Display the DataFrame to check the new column
    print(df_Yield.head())

    # List the headings (column names)
    headings = df_Yield.columns.tolist()

    # Print the headings
    print("DataFrame Headings:")
    for heading in headings:
        print(heading)

    states = ['Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal', 'Puducherry', 'Goa', 'Andhra Pradesh',
              'Tamil Nadu', 'Odisha', 'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram', 'Punjab',
              'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand',
              'Jharkhand', 'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana', 'Arunachal Pradesh', 'Sikkim']

    pattern = '|'.join(states)
    df_Rain_filtered = df_Rain[df_Rain['SUBDIVISION'].str.contains(pattern, case=False, na=False)].copy()
    print("Filtered Rain DataFrame:")
    print(df_Rain_filtered)

    # Standardize columns
    df_Rain_filtered['SUBDIVISION'] = df_Rain_filtered['SUBDIVISION'].str.strip().str.lower()

    df_Yield['State'] = df_Yield['State'].str.strip().str.lower()

    state_to_subdivision = {state: process.extractOne(state, df_Rain_filtered['SUBDIVISION'].unique())[0] for state in
                            df_Yield['State'].unique()}
    df_Yield['Matched_Subdivision'] = df_Yield['State'].map(state_to_subdivision)

    df_Yield = df_Yield.merge(df_Rain_filtered, left_on=['Matched_Subdivision', 'Crop_Year'],
                              right_on=['SUBDIVISION', 'YEAR'], how='left')
    print("Merged DataFrame with Filtered Rain Data:")
    print(df_Yield.head())

    headings = df_Yield.columns.tolist()
    print("DataFrame Headings:")
    for heading in headings:
        print(heading)

    df_Yield = combine_monthly_data(df_Yield)

    # Convert the strings to lists safely
    df_Yield['Rain_Monthly_Data'] = df_Yield['Rain_Monthly_Data'].apply(safe_eval)
    df_Yield['Months'] = df_Yield['Months'].apply(safe_eval)

    # Dictionary to map months to their index
    month_to_index = {
        'January': 0, 'February': 1, 'March': 2, 'April': 3,
        'May': 4, 'June': 5, 'July': 6, 'August': 7,
        'September': 8, 'October': 9, 'November': 10, 'December': 11, 'Annual': 12
    }

    def calculate_monthly_average(rain_data, months):
        indices = [month_to_index[month] for month in months]
        values = [rain_data[i] for i in indices]
        return np.mean(values)

    df_Yield['Average_Rainfall'] = df_Yield.apply(lambda row: calculate_monthly_average(row['Rain_Monthly_Data'], row['Months']),
                                      axis=1)
    df_Yield.to_csv('output.csv', index=False)

    # Preprocess the data
    X = df_Recommendation.drop('label', axis=1)  # Features
    y = df_Recommendation['label']  # Target

    # Encode the target variable if it's categorical
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_dist = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500],
        'max_depth': [-1, 10, 20],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'bagging_freq': [0, 5, 10],
        'verbosity': [-1]
    }

    # Initialize the LightGBM model
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        num_class=len(label_encoder.classes_),
        metric='multi_logloss',
        verbosity=-1
    )

    # Use RandomizedSearchCV to find the best hyperparameters
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1,
                                       verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)
    best_model = random_search.best_estimator_

    # Predict using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save the model and label encoder
    save_model_and_encoder(best_model, label_encoder)

    # Feature Importance
    feature_importance = best_model.feature_importances_
    features = X.columns

    # Create a DataFrame for the feature importance
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

    # Example usage of predict_best_crop
    input_params = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.5,
        'humidity': 80,
        'ph': 6.5,
        'rainfall': 200
    }

    best_crop = predict_best_crop(best_model, label_encoder, input_params)
    print(f"The best crop to plant is: {best_crop}")

    # Filter `df_Yield` to only include rows where the "Crop" matches the predicted crop
    df_filtered = df_Yield[df_Yield['Crop'] == best_crop]
    df_filtered = df_Yield[df_Yield['State'] == "andhra pradesh"]
    # Display the filtered DataFrame
    print("Filtered DataFrame:")
    print(df_filtered)
    df_filtered.to_csv('Filtered.csv', index=False)

    # Drop the specified columns
    df_Input = df_filtered.drop(columns=['Production', 'YEAR', 'Months' ,'Rain_Monthly_Data','Season','State','Matched_Subdivision','SUBDIVISION','Crop_Year','Crop'])

    # Save the resulting DataFrame to a new CSV file
    output_file_path = 'filtered_crop_data.csv'
    df_Input.to_csv(output_file_path, index=False)

    print(f"Filtered DataFrame saved to {output_file_path}")

    df_Input = df_Input.dropna(subset=['Average_Rainfall'])


    # Separate features and target
    X = df_Input.drop(columns=['Yield'])
    y = df_Input['Yield']

    correlation_matrix = df_Input.corr()
    target_variable = 'Yield'
    feature_correlation = correlation_matrix[target_variable].drop(target_variable)
    print(feature_correlation)

    feature_correlation = correlation_matrix[target_variable].drop(target_variable)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)


    model = RandomForestRegressor(n_estimators=11)
    model.fit(x_train, y_train)
    rf_predict = model.predict(x_test)
    print(rf_predict)
    print(model.score(x_test,y_test))
    r1 = r2_score(y_test, rf_predict)
    print("R2 score : ", r1)

    # Creating the first KDE plot
    ax = sns.kdeplot(y_test, color="r", label="Actual value")
    # Adding the second KDE plot to the same axes
    sns.kdeplot(rf_predict, color="b", label="Predicted Values", ax=ax)
    # Adding a title to the plot
    plt.title('Random Forest Regression')
    # Displaying the plot
    plt.show()


if __name__ == "__main__":
    main()
