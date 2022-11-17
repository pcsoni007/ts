from flask import Flask
from flask_cors import CORS
from flask_caching import Cache
# from flask_redis import FlaskRedis
from dotenv import load_dotenv
load_dotenv()
import os


config = {
    "DEBUG": True,                  # some Flask specific configs
    "CACHE_TYPE": "RedisCache",    # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 3000,    # run app in debug mode
    "CACHE_REDIS_URL": os.getenv('REDIS_URL'),
    "CACHE_REDIS_PASSWORD": os.getenv('REDIS_PASS'),
    "CACHE_REDIS_PORT": os.getenv('REDIS_PORT')

}

# Flask to use the above defined config
app = Flask(__name__)
cors = CORS(app)
# cache = FlaskRedis(app)

app.config.from_mapping(config)
cache = Cache(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'somesecretkeythatonlyishouldknow'

from controllers import delete_files
from controllers import controller
from controllers import data_preprocessing

# set FLASK_ENV = development
# if __name__ == '__main__':
#     app.run(port=5000)