from flask import request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import models.timeseries as timeseries
# import models.timeseries as timeseries
import os
from app import app
from time import time
import sys
import json
sys.setrecursionlimit(1500)


ALLOWED_EXTENSIONS = set(['xlsx', 'csv'])

# Every new unique file name will be mapped with the dataframe as a key-value pair and stored inside df_mapping dict
df_mapping = {}


def convert_json_to_df(data):
    df = pd.DataFrame(data)
    print(df.transpose())


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_metadata(filename, getOnlyDataFrame=False):
    df = None

    if ('xlsx' in filename):
        df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    elif ('csv' in filename):
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if getOnlyDataFrame:
        return df

    # rows, cols = timeseries.shape_df(df)
    rows_cols = timeseries.Shape_df(df)
    desc = (timeseries.data_description(df))
    head = (timeseries.display_head(df))
    tail = (timeseries.display_tail(df))

    return rows_cols, desc, head, tail, df


def get_unique_name(ext=''):
    milliseconds = int(time() * 1000)
    return str(milliseconds) + '.' + ext if ext else str(milliseconds)


@app.post('/upload')
def upload_file():

    if 'file' not in request.files:
        return jsonify({"message": "Bad request, Missing file attribute in files."}), 400

    else:
        file = request.files['file']
        new_unique_name = None

        if file.filename == '':
            return jsonify({"message": "Bad request, please provide a file."}), 400

        if file and allowed_file(file.filename):
            try:
                extension = file.filename.split('.')[-1]
                new_unique_name = get_unique_name(extension)
                filename = secure_filename(new_unique_name)
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], new_unique_name))

                rows_cols, desc, head, tail, df = get_metadata(filename)

                df_mapping[new_unique_name] = df

                print(df_mapping)

                return jsonify({
                    "shape": json.dumps(rows_cols),
                    "description": desc.to_json(default_handler=str, orient='columns'),
                    "head": head.to_json(orient='records'),
                    "unique_filename": new_unique_name
                }), 201
                # return jsonify({
                #     "shape": json.dump(rows_cols),
                #     "description": json.dump(desc),
                #     "tail": json.dump(tail),
                #     "head": json.dump(head),
                #     "unique_filename": new_unique_name
                # }), 201

            except Exception as ex:
                print(ex)
                return jsonify({"message": "An error occured while uploading the file."}), 500

        else:
            return jsonify({"message": "Invalid file format. Required csv or xlsx extensions."}), 400


@app.get('/uploads/<df_key>')
def uploaded_file(df_key):
    print(df_key)
    if df_key in df_mapping.keys():
        try:
            rows_cols, desc, head, tail, df = get_metadata(df_key)

            return jsonify({
                "message": "Success",
                "shape": rows_cols.to_json(orient="values"),
                "description": desc.to_json(orient='columns'),
                "tail": tail.to_json(orient='records'),
                "head": head.to_json(orient='records'),
                "unique_filename": df_key
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "An error occured while uploading the file."}), 500
    else:
        return jsonify({"message": "Missing df_key " + df_key}), 404


# Send file name in file_name as a query param to this URI
@app.get('/summary')
def summary():

    # filename = request.args['file_name'] if 'file_name' in request.args else None
    # df = get_metadata(filename, True)

    # global column, y, Name, x, type

    # # Todo: Check with ML team, what this graph() does
    # x = timeseries.graph(df)
    # y = timeseries.null_list(df)

    # Name = request.form['column']
    # type = request.form['row']

    # timeseries.drop_rows(df, Name)

    # l = timeseries.null_list(df)
    # k = timeseries.null_list(df)

    return jsonify({"message": "Processed"})

    #     return render_template('summary.html', data3=l, data2=k)
    # return render_template('summary.html', data1=x, data2=y)


@app.get('/eda/<filename>')
def eda(filename):
    pass
    # return render_template('EDA.html', data=column)


@app.get('/stationarity/<filename>')
def stationarity(filename):
    pass
    # return render_template('stationarity.html', data=column)


@app.get('/tsplot/<filename>')
def tsplot(filename):
    pass
    # return render_template('TsPlots.html', data=column)


@app.get('/prediction/<filename>')
def prediction(filename):
    pass
    # return render_template('prediction.html')
