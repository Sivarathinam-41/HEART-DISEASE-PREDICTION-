import os
from flask import Flask, request, render_template, jsonify
from heart_disease_model import (
    predict_heart_disease,
    get_metrics,
    get_demo_presets,
    generate_clinical_recommendations,
    get_cardio_chat_response,
    FEATURE_ORDER
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static'
)

@app.route('/')
def home():
    metrics = get_metrics()
    presets = get_demo_presets()
    return render_template(
        'index.html',
        metrics=metrics,
        presets=presets,
        active_preset='default',
        prediction_result=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features deterministically using explicit feature names
        form_data = {}
        for feature in FEATURE_ORDER:
            val = request.form.get(feature)
            if val is None or val.strip() == '':
                raise ValueError(f"Missing required parameter: '{feature}'")
            form_data[feature] = float(val)

        result = predict_heart_disease(form_data)
        metrics = get_metrics()
        presets = get_demo_presets()

        return render_template(
            'index.html',
            prediction_result=result,
            form_values=form_data,
            metrics=metrics,
            presets=presets
        )

    except Exception as e:
        metrics = get_metrics()
        presets = get_demo_presets()
        return render_template(
            'index.html',
            error_message=str(e),
            form_values=request.form.to_dict(),
            metrics=metrics,
            presets=presets
        )

# ==========================================================
# REST API Endpoints (For decoupled Vercel / mobile frontends)
# ==========================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for model inference with automated clinical recommendations."""
    try:
        data = request.get_json(force=True, silent=False)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON payload provided'}), 400

        result = predict_heart_disease(data)
        recommendations = generate_clinical_recommendations(data, result)
        result['recommendations'] = recommendations

        return jsonify({
            'success': True,
            'data': result
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """CardioBot AI assistant conversational endpoint."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        message = payload.get('message', '')
        patient_data = payload.get('patient_data', {})
        risk_result = payload.get('risk_result', {})

        if not message.strip():
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400

        reply = get_cardio_chat_response(message, patient_data, risk_result)
        return jsonify({
            'success': True,
            'response': reply
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """Endpoint for generating personalized lifestyle and medical recommendations."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        patient_data = payload.get('patient_data', {})
        risk_result = payload.get('risk_result', {})

        recs = generate_clinical_recommendations(patient_data, risk_result)
        return jsonify({
            'success': True,
            'data': recs
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return model cross-validation and test set evaluation metrics."""
    return jsonify({
        'success': True,
        'metrics': get_metrics()
    }), 200

@app.route('/api/presets', methods=['GET'])
def api_presets():
    """Return pre-configured patient profiles for one-click testing."""
    return jsonify({
        'success': True,
        'presets': get_demo_presets()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
