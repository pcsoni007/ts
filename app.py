from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'somesecretkeythatonlyishouldknow'

from controllers import delete_files
from controllers import controller
from controllers import data_preprocessing
