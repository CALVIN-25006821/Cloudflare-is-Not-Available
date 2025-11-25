from flask import Flask, request, jsonify, render_template
from risk_engine import get_total_risk

app = Flask(__name__)

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for risk prediction"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract parameters
        age = int(data.get('age', 30))
        income = float(data.get('income', 5000))
        loan_amount = float(data.get('loan_amount', 10000))
        loan_term = int(data.get('loan_term', 12))
        dti = float(data.get('dti', 0.4))
        credit_history = int(data.get('credit_history', 5))
        dependents = int(data.get('dependents', 1))
        user_story = data.get('user_story', '')
        
        # Calculate risk
        result = get_total_risk(
            age, income, loan_amount, loan_term, 
            dti, credit_history, dependents, user_story
        )
        
        # Add recommendation
        if result['Final_Risk'] > 50:
            result['Recommendation'] = 'REJECT LOAN'
            result['Recommendation_Class'] = 'danger'
        else:
            result['Recommendation'] = 'APPROVE LOAN'
            result['Recommendation_Class'] = 'success'
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask application on http://127.0.0.1:5000/")
    print("📊 Make sure baseline_model_rf.pkl exists (run train_model.py if needed)")
    print("🔑 Set GEMINI_API_KEY environment variable for AI text analysis")
    app.run(debug=True, port=5000)