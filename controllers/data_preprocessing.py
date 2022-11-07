from flask import request, jsonify
import models.timeseries as timeseries
from app import app
from controllers.controller import df_mapping, get_metadata
import pandas as pd
import json
import numpy as np


@app.get('/null/graph')
def null_page():

    if (request.headers['unique_filename']):

        file_name_from_headers = request.headers['unique_filename']
        # col_name = request.headers['col']
        if file_name_from_headers in df_mapping.keys():

            try:

                df = df_mapping[file_name_from_headers]
                graph = timeseries.graph(df)
                null_percent = timeseries.null_list(df)

                return jsonify({
                    "message": "Success",
                    "graph": graph,
                    "nulls_with_percentage": null_percent
                }), 200

            except Exception as ex:
                print(ex)
                return jsonify({"message": "Something went wrong."}), 500

        else:
            return jsonify({"message": "Missing filename " + file_name_from_headers}), 400

    else:
        return jsonify({"message": "Missing filename in request" + file_name_from_headers}), 400


@app.get('/null/action')
def null_action():
    if (request.headers['unique_filename']) and \
        (request.headers['col']) and \
            (request.headers['action']):

        filename = request.headers['unique_filename']
        col_name = request.headers['col']
        action = request.headers['action']

        if filename in df_mapping.keys():
            global df
            df = df_mapping[filename]
            try:
                if action == 'impute' and (request.headers['impute_method']):
                    impute_method = request.headers['impute_method']
                    timeseries.impute(
                        df, col_name, (impute_method if impute_method else ''))

                elif action == 'drop_rows':
                    timeseries.drop_rows(df, col_name)

                elif action == 'drop_cols':
                    timeseries.drop_cols(df, col_name)

                else:
                    return jsonify({'message': 'Please choose correct action or impute method.'}), 400

                return jsonify({"payload": df.to_json(orient='records')}), 200

            except Exception as ex:
                print(ex)
                return jsonify({"message": "Something went wrong."}), 500

        else:
            return jsonify({"message": "Missing filename, please upload the file again."}), 400

    else:
        return jsonify({"message": "All the headers are required"}), 400


# Resampling Page

@app.get('/intervals')
def exhibit_freq():
    if (request.headers['date_col']):

        date_col = request.headers['date_col']

        try:
            df_date = timeseries.dtype_conversion(df, date_col)
            intervals = timeseries.exhhibitFreq(
                df_date, date_col)
            return jsonify({
                "message": "Success",
                "frequencies": intervals
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Missing filename, please upload the file again."}), 400


@app.get('/resampling')
def resampling_data():
    if (request.headers['date_col']) and \
            (request.headers['period']):

        date_col = request.headers['date_col']
        period = request.headers['period']

        try:
            global final_data
            df_con = timeseries.dtype_conversion(df, date_col)
            resampling_data = timeseries.resamplingData1(
                df_con, period, date_col)
            print("Resampled Data")
            print(resampling_data)
            set_res_data = resampling_data.copy()
            print("Copied data")
            print(set_res_data)
            final_data = timeseries.set_index(set_res_data, date_col)
            print("Final Data")
            print(final_data)
            return jsonify({
                "message": "Success",
                "resampling_data": resampling_data.to_json(orient='records'),
                "col_name": date_col
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Missing filename, please upload the file again."}), 400


# Stationarity Page

@app.get('/stationarity')
def stationarity_data():
    print(final_data)
    if (request.headers['tar_col']):
        global target_col
        tar_col = request.headers['tar_col']
        target_col = final_data[tar_col]
        print(target_col)
        try:
            stationarity = timeseries.stationarity_test3(
                final_data, target_col)

            return jsonify({
                "message": "Success",
                "stationarity check": json.dumps(stationarity, default=default)
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.get('/Autocorrelation')
def acf_and_pacf():
    if (request.headers['tar_col']):
        tar_col = request.headers['tar_col']
        try:
            graph = timeseries.acf_pacf(final_data[tar_col])

            return jsonify({
                "message": "Success",
                "acf and pacf": json.dumps(graph, default=default)
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500
