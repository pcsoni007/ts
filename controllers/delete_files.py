import os
import datetime
from app import app
from flask import request, jsonify


@app.delete('/files')
def delete_files():
    path = app.config['UPLOAD_FOLDER']
    list = []
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.csv' or '.xlsx' in f:
                print(f)
                c_timestamp = os.path.getctime("/".join([path, f]))
                creation_time = datetime.datetime.fromtimestamp(c_timestamp)
                current_time = datetime.datetime.now()
                NUMBER_OF_SECONDS = 86400
                if (current_time - creation_time).total_seconds() > NUMBER_OF_SECONDS:
                    os.remove("/".join([path, f]))
    return jsonify({
        "message": "Deleted successfully"
    }), 200
