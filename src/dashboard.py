from flask import Flask, render_template, jsonify
import pandas as pd
app = Flask(__name__)


# Mock data - replace with actual data retrieval logic
def get_performance_data():
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-07-01', freq='D'),
        'incidents': np.random.randint(0, 100, size=182),
        'accuracy': np.random.uniform(0.7, 1.0, size=182)
    })


def get_fraud_types():
    return {
        'Account Takeover': 30,
        'Payment Fraud': 25,
        'Identity Theft': 20,
        'Phishing': 15,
        'Other': 10
    }


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/performance_data')
def performance_data():
    df = get_performance_data()
    return jsonify({
        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'incidents': df['incidents'].tolist(),
        'accuracy': df['accuracy'].tolist()
    })


@app.route('/fraud_types')
def fraud_types():
    return jsonify(get_fraud_types())


if __name__ == '__main__':
    app.run(debug=True)
