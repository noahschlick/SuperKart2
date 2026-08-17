# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning pipeline (preprocessing + tuned Random Forest model)
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    Handles GET requests to the root URL ('/') of the API.
    Returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint for single product/store sales prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    """
    Handles POST requests to the '/v1/predict' endpoint.
    Expects a JSON payload containing product and store details and returns
    the predicted total sales for that product at that store.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract the features expected by the model, using the same 10-column
    # schema the model was trained on
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category'],
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make the prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert to a native Python float so jsonify can serialize it
    # (model.predict returns a NumPy float, which jsonify cannot encode directly)
    predicted_sales = round(float(predicted_sales), 2)

    # Return the prediction
    return jsonify({'Predicted_Product_Store_Sales_Total': predicted_sales})


# Define an endpoint for batch sales prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Handles POST requests to the '/v1/predictbatch' endpoint.
    Expects a CSV file containing product and store details for multiple
    product/store combinations and returns the predicted sales for each
    row as a dictionary, keyed by row index.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for every row in the DataFrame
    predicted_sales = model.predict(input_data).tolist()

    # Round the predictions for readability
    predicted_sales = [round(float(pred), 2) for pred in predicted_sales]

    # Build a dictionary of predictions keyed by row index
    output_dict = dict(zip(input_data.index.astype(str), predicted_sales))

    # Return the predictions dictionary as a JSON response
    return output_dict


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
