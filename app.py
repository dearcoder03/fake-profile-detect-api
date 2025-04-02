import joblib
from flask import Flask, jsonify, request
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "fraud_detection_model.joblib")
model = joblib.load(model_path)

@app.route('/fraud_result', methods=['POST'])
def fraud_result():
    try:
        data = request.get_json()
        
        totalWeight = data.get('noOfPosts', 0)
        captionData = data.get('captionData', [])
        bioText = data.get('bioText', {})
        
        fraud_result = 0

        # Process captions
        for i in range(min(totalWeight, len(captionData))):
            currCaptionData = captionData[i]
            
            if 'Caption' in currCaptionData and currCaptionData['Caption']:
                currText = currCaptionData['Caption'].lower()
                prediction = model.predict([currText])
                fraud_result += prediction[0]
        
        # Process bioText
        if isinstance(bioText, dict) and 'Caption' in bioText and bioText['Caption']:
            bioTextContent = bioText['Caption'].lower()
            prediction = model.predict([bioTextContent])
            fraud_result += prediction[0]
        
        # Calculate fraud percentage
        fraud_percent = (fraud_result / (totalWeight + 1)) * 100 if totalWeight > 0 else 0
        
        return jsonify({'fake_value': f"{fraud_percent:.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
