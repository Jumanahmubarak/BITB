from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

app = Flask(__name__)

# Train the model and save it
def train_model():
    # Load data from CSV file
    data = pd.read_csv("bib_attack_data.csv")
    
    # Data preprocessing
    label_encoder = LabelEncoder()
    text_columns = ['font_style', 'image_quality', 'button_spacing', 'font_resolution', 
                    'button_position', 'button_size', 'text_type', 'text_formatting']
    
    # Process text columns
    for col in text_columns:
        # Ensure values don't contain spaces or unsupported values
        data[col] = data[col].apply(lambda x: str(x).strip() if isinstance(x, str) else 'Unknown')
        data[col] = label_encoder.fit_transform(data[col])

    # Allowed values for `button_text`
    allowed_values = [
        "Sign In", "Login", "Log In", "Reset Password",
        "Verify Account", "Account Security", "Account Recovery", "Billing Information"
    ]

    # Encode allowed values using LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(allowed_values)
    # Process columns like `button_text` and `suspicious_terms`
    data['button_text'] = data['button_text'].apply(lambda x: x if x in allowed_values else 'Unknown')
    data['button_text'] = label_encoder.transform(data['button_text'])

    # Allowed terms for `suspicious_terms`
    allowed_terms = [
        "account recovery", "verify account", "reset password", 
        "billing information", "N/A"
    ]

    # Process `suspicious_terms`
    data['suspicious_terms'] = data['suspicious_terms'].apply(lambda x: x.strip().lower() if isinstance(x, str) else "n/a")
    data['suspicious_terms'] = data['suspicious_terms'].apply(lambda x: x if x in allowed_terms else "unknown")
    
    # Encode values using LabelEncoder
    label_encoder_suspicious_terms = LabelEncoder()
    data['suspicious_terms'] = label_encoder_suspicious_terms.fit_transform(data['suspicious_terms'])

    # Process time-related columns like "1s" or "2.5s" and convert them to numbers
    time_columns = ['button_click_time', 'page_load_time', 'text_interaction_time']

    for col in time_columns:
        if col in data.columns:  # Ensure the column exists
            # Remove the 's' character if present
            data[col] = data[col].apply(lambda x: str(x).replace('s', '').strip() if isinstance(x, str) else x)

            # Convert values to float with error handling
            data[col] = pd.to_numeric(data[col], errors='coerce')

            # Replace invalid values (NaN) with a default value like 0
            data[col] = data[col].fillna(0)

    # Process columns with values like 'High', 'Low', or 'Medium'
    binary_columns = ['image_quality']  # Example for binary columns like 'High' and 'Low'
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: 1 if x == 'High' else 0 if x == 'Low' else -1).astype(float)

    # Process `window_size` column
    data = data[data['window_size'].str.contains('x', na=False)]
    data[['window_width', 'window_height']] = data['window_size'].str.split('x', expand=True).astype(int)
    
    # Drop irrelevant columns
    data = data.dropna()
    X = data.drop(['URL', 'is_bib_attack', 'window_size'], axis=1)
    y = data['is_bib_attack']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Save column order to ensure consistent order during prediction
    column_order = X_train.columns

    # Train the model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model and column order
    with open("trained_model.pkl", "wb") as file:
        pickle.dump((model, column_order, label_encoder, label_encoder_suspicious_terms), file)
    print("Model trained and saved successfully!")

# Load the trained model
def load_model():
    with open("trained_model.pkl", "rb") as file:
        model, column_order, label_encoder, label_encoder_suspicious_terms = pickle.load(file)
    return model, column_order, label_encoder, label_encoder_suspicious_terms 

# Train the model (run only once)
train_model()

# Load the trained model
model, column_order, label_encoder, label_encoder_suspicious_terms = load_model()

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Preprocess input data in the same way as training data
        input_data = pd.DataFrame([data])

        # Remove `URL` column from input data
        input_data = input_data.drop(['URL'], axis=1, errors='ignore')

        # Ensure columns used in training are present in input data
        label_encoder = LabelEncoder()
        text_columns = ['font_style', 'image_quality', 'button_spacing', 'font_resolution', 
                        'button_position', 'button_size', 'text_type', 'text_formatting']
        for col in text_columns:
            input_data[col] = input_data[col].apply(lambda x: str(x).strip() if isinstance(x, str) else 'Unknown')
            input_data[col] = label_encoder.fit_transform(input_data[col])

        # Process values like 'High' and 'Low' in columns like `image_quality`
        binary_columns = ['image_quality']
        for col in binary_columns:
            input_data[col] = input_data[col].apply(lambda x: 1 if x == 'High' else 0 if x == 'Low' else -1).astype(float)

        if 'suspicious_terms' in input_data.columns:
            input_data['suspicious_terms'] = input_data['suspicious_terms'].apply(lambda x: x.strip().lower() if isinstance(x, str) else "n/a")
            input_data['suspicious_terms'] = input_data['suspicious_terms'].apply(lambda x: x if x in label_encoder_suspicious_terms.classes_ else "unknown")
        
            # Add "Unknown" if not already in allowed classes
            if "unknown" not in label_encoder_suspicious_terms.classes_:
                label_encoder_suspicious_terms.classes_ = np.append(label_encoder_suspicious_terms.classes_, "unknown")
        
            # Convert text to numeric
            input_data['suspicious_terms'] = label_encoder_suspicious_terms.transform(input_data['suspicious_terms'])
        else:
            # If the column is missing, add it with default values
            input_data['suspicious_terms'] = 0

        # Handle `window_size` column
        if 'window_size' in input_data.columns:
            input_data['window_width'], input_data['window_height'] = zip(*input_data['window_size'].str.split('x').apply(lambda x: (int(x[0]), int(x[1]))))
            input_data = input_data.drop(['window_size'], axis=1)
        
        input_data['button_text'] = input_data['button_text'].apply(lambda x: x if x in label_encoder.classes_ else 'Unknown')
        if 'Unknown' not in label_encoder.classes_:
            label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')  # Add "Unknown" if not already present
        input_data['button_text'] = label_encoder.transform(input_data['button_text'])

        # Process time columns
        time_columns = ['button_click_time', 'page_load_time', 'text_interaction_time']
        for col in time_columns:
            if col in input_data.columns:  # Ensure the column exists
                # Remove the 's' character if present
                input_data[col] = input_data[col].apply(lambda x: str(x).replace('s', '').strip() if isinstance(x, str) else x)

                # Convert values to float with error handling
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

                # Replace invalid values (NaN) with the default value 0
                input_data[col] = input_data[col].fillna(0)

        # Add missing columns with default values
        for col in column_order:
            if col not in input_data.columns:
                input_data[col] = 0  # Set default values like 0 if the column is missing

        # Arrange columns in the same order as during training
        input_data = input_data[column_order]

        # Make prediction
        prediction = model.predict(input_data)
        result = "Phishing" if prediction[0] == 1 else "Legitimate"
        if result == "Phishing":
            suggestions = [
                "Check the website's URL.",
                "Avoid entering sensitive information on this page.",
                "Verify the legitimacy of the website through official channels."
                ];
        else :
            suggestions = ["This page appears to be safe to use."];
        

        return jsonify({"message": f"The page is {result}!", "suggestions": suggestions})
    
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)













