from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS
from joblib import load

# LOAD MODEL
model = load('decision_tree_best_model.pkl')

# ENCODING PURPOSES
encoders = {
    "Item_Type": [
        'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',
        'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household',
        'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
    ],
    "Outlet_Identifier": [
        'OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035',
        'OUT045', 'OUT046', 'OUT049'
    ],
    "Outlet_Location_Type": ['Tier 1', 'Tier 2', 'Tier 3'],
    "Outlet_Type": [
        'Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'
    ],
}

# TO KNOW WHICH FIELD TO ENCODE
encoded_fields = ["Item_Type", "Outlet_Location_Type", "Outlet_Type"]

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate that 'category' and 'location' exist and have data
        if 'category' not in data or not data['category'] or 'location' not in data:
            return jsonify({"error": "Missing or empty 'category' or 'location' field"}), 250

        # Initialize input data
        input_data = []

        # Feature 1: Average visibility
        avg_visibility = 0.1 / len(data['category'])
        visibility_feature = [avg_visibility for _ in range(len(data['category']))]

        # Feature 2: Encoded category
        category_feature = [
            encoders['Item_Type'].index(item) for item in data['category']
        ]

        # Feature 3: Location (assuming it's a single value repeated)
        feature_3 = [data['location'] for _ in range(len(data['category']))]

        # Feature 4: Category count bins
        if len(data['category']) < 10:
            feature_4 = [0 for _ in range(len(data['category']))]
        else:
            feature_4 = [data['location'] + 1 for _ in range(len(data['category']))]

        # Combine all features for each sample
        for i in range(len(data['category'])):
            input_data.append([
                visibility_feature[i],
                category_feature[i],
                feature_3[i],
                feature_4[i]
            ])

        # Convert to numpy array
        input_data = np.array(input_data)

        # Predict probabilities
        probabilities = model.predict_proba(input_data)

        # Calculate mean probabilities across all samples
        mean_probabilities = np.mean(probabilities, axis=0)

        # Prepare summary prediction with percentages
        summary_prediction = {
            encoders['Outlet_Identifier'][idx]: round(mean_probabilities[idx] * 100, 2)
            for idx in range(len(mean_probabilities))
        }

        return jsonify({"summary_prediction": summary_prediction}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)