from crypt import methods
import logging
import predict
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model_path = 'models/model.h5'
model = load_model(model_path)

# Predict
prediction = predict.test(model, 28, 28)

# Display the homepage
@app.route('/')
def root():
    app.logger.info("Prediction: " + str(prediction))
    return render_template('index.html', pred=prediction)

# Handle requests to predict values
@app.route('/predict')
def pred():
    app.logger.info("Recieved: " + request.args.to_dict().keys())
    prediction = "Null"
    try:
        app.logger.info("Converted:" + str(request.args.to_dict()['Base64String'][0:4]))
        base64_string = request.args.to_dict()['Base64String'][22:]
        prediction = str(predict.predict_from_string(base64_string, model, 28, 28))
    except KeyError:
        pass
    return prediction

if __name__ == '__main__':
    # Used only when running locally
    app.run(host='127.0.0.1', port=8080, debug=True)

if __name__ != '__main__':
    # Add gunicorn logging functionality
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)