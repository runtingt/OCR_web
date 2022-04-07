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
@app.route('/', methods=['GET'])
def root():
    app.logger.info("Prediction: " + str(prediction))
    app.logger.info("Recieved: " + str(request.data))
    app.logger.info("Args: " + str(request.args))
    app.logger.info("Converted:" + str(request.args.to_dict()['Base64String'][0:2]))
    return render_template('index.html', pred=prediction)

if __name__ == '__main__':
    # Used only when running locally
    app.run(host='127.0.0.1', port=8080, debug=True)

if __name__ != '__main__':
    # Add gunicorn logging functionality
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)