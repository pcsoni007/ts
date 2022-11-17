from flask import request, jsonify
import models.timeseries as timeseries
from app import app
from controllers.controller import df_mapping, get_metadata
import pandas as pd
import json
import numpy as np
from app import cache
import plotly.offline as pyo

# import plotly.io as pio

# cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

@app.get('/null/graph')
def null_page():

    if (request.headers['unique_filename']):

        file_name_from_headers = request.headers['unique_filename']
        # col_name = request.headers['col']
        if file_name_from_headers in df_mapping.keys():

            try:

                df = cache.get('df')#df_mapping[file_name_from_headers]
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

import pdb

@app.get('/null/action')
def null_action():
    # pdb.set_trace()
    # print("Perform" + request.headers['action'] +"action on")

    if (request.headers['unique_filename']) and \
        (request.headers['col']) and \
            (request.headers['action']):
        # print("Perform" + request.headers['action'] +"action on")

        filename = request.headers['unique_filename']
        col_name = request.headers['col']
        action = request.headers['action']

        if filename in df_mapping.keys():
            # global df
            df = cache.get('df')
            try:
                if action == 'impute': #and (request.headers['impute_method']):
                    #impute_method = request.headers['impute_method']
                    # print("Impute executing in dataprove")
                    # print('Before impute', df)
                    timeseries.impute(df, col_name)#, (impute_method if impute_method else ''))
                    # print('after impute', df)
                elif action == 'drop_rows':
                    timeseries.drop_rows(df, col_name)

                elif action == 'drop_cols':
                    timeseries.drop_cols(df, col_name)

                else:
                    return jsonify({'message': 'Please choose correct action or impute method.'}), 400
                payload = df.to_json(orient='records')
                cache.set('df',df)
                print('after null action', df)
                print('Payload', payload)
                return jsonify({"payload": payload}), 200

            except Exception as ex:
                print(ex)
                return jsonify({"message": "Something went wrong."}), 500

        else:
            return jsonify({"message": "Missing filename, please upload the file again."}), 400

    else:
        print("some error occurred")
        return jsonify({"message": "All the headers are required"}), 400


# Resampling Page

@app.get('/intervals')
def exhibit_freq():
    if (request.headers['date_col']):

        date_col = request.headers['date_col']
        date_format = request.headers['date_format']
        print('DataCol',date_col)
        print('DataFormat',date_format)
        df = cache.get('df')
        try:
            df_date = timeseries.dtype_conversion(df, date_col)
            intervals = timeseries.exhhibitFreq(df, date_col, date_format)
            print("Frequency of dataset",intervals)

            return jsonify({
                "message": "Success",
                "frequency": intervals
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Missing filename, please upload the file again."}), 400


@app.get('/resampling')
def resampling_data():
    if (request.headers['date_col']) and \
            request.headers['date_format'] and \
            (request.headers['period']):

        date_col = request.headers['date_col']
        period = request.headers['period']
        dtformat = request.headers['date_format']
        print(date_col, period, dtformat)
        try:
            global final_data
            df = cache.get('df')
            resampled_data = timeseries.resamplingData(df, period, date_col, dtformat)
            print("Resampled Data")
            print(resampled_data)
            cache.set('df1', resampled_data)
            set_res_data = resampled_data.copy()
            print("Copied data")
            print(set_res_data)
            tempDf = resampled_data.reset_index()
            tempDf[date_col] =  resampled_data.reset_index()[date_col].dt.strftime('%d-%m-%Y')
            print('Resampled Data', tempDf.to_json(orient='records'))
            return jsonify({
                "message": "Success",
                "resampled_data": tempDf.to_json(orient='records'),
                "col_name": date_col,
                "resampled_df": tempDf.to_json(orient='split')
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Missing filename, please upload the file again."}), 400


########################  EDA Page  #############################
@app.get('/datevstargetcol')
def plot_vs_graph():
    global target_col
    target_col = request.headers['target_col']
    if (target_col):
        final_data = cache.get('df1')
        # tar_col = request.headers['target_col']
        try:
            vs_graph = timeseries.plot_vs_graph(final_data, target_col)
            print('VS graph \n', vs_graph)
            return jsonify({
                "message": "Success",
                "vs_graph": vs_graph
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Please select the Target Column!"}), 400


@app.get('/resampledplot')
def plot_resample():
    alias = request.headers['alias']
    if (alias):
        try:
            df = cache.get('df1')
            plot_col = timeseries.resample_plot(df, target_col, alias)
    
            return jsonify({
                "message": "Success",
                "resampled_plot": plot_col
            }), 200

        except Exception as ex:

            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Please Enter the period alias for Resampling Plot "}), 400



@app.get('/topnplot')
def plot_top_values():

    if (request.headers['topn']):
        top_n = request.headers['topn']

        try:
            df = cache.get('df1')
            plot_col = timeseries.plot_top_n(df, target_col, top_n)

            return jsonify({
                "message": "Success",
                "top_n_plot": plot_col

            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500

    else:
        return jsonify({"message": "Please Enter the Value of n for Top n values Plot "}), 400


# Stationarity Page

@app.get('/decompose')
def decomposition():

    target_col = request.headers['target_col']
    cache.set('target_col', target_col)
    decomposition_type = request.headers['decomposition_type']
    print(target_col, decomposition_type)
    final_data = cache.get('df1')

    print(final_data)
    if (target_col):
        try:
            decomposition_plots = timeseries.decomposition(final_data[target_col], decomposition_type)
            # print('Decompostion plots', decomposition_plots)
            return jsonify({
                "message": "Success",
                'interpretation': decomposition_plots['interpretation'],
                'trend_component': decomposition_plots['trend_component'],
                'seasonal_component': decomposition_plots['seasonal_component'],
                'cyclical_component': decomposition_plots['cyclical_component'],
                'irregular_component': decomposition_plots['irregular_component'],
                'note': decomposition_plots['note'],
                "observed_graph" : decomposition_plots['observed_graph'],
                "trend_graph" : decomposition_plots['trend_graph'],
                'seasonal_graph' : decomposition_plots['seasonal_graph'],
                'residual_graph' : decomposition_plots['residual_graph'],
                # "decomposition_plots": json.dumps(decomposition_plots, default=default)
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500


@app.get('/stationaritycheck')
def stationarity_plot():

    target_col = cache.get('target_col')
    df = cache.get('df1')

    if (target_col):
        try:
            stationarity = timeseries.stationarity_check_dict(df, target_col)

            return jsonify({
                "message": "Success",
                "stationarity": " Stationarity means that the statistical properties of a process generating a time series do not change over time. That is Mean and Standard deviation is approximately constant over time.\n\nStationarity Graph represents stationarity of the series w.r.t. Time. X axis depicts time and Y axis depicts Dependent variable . Blue line represents the original Time series data , Red line represents Mean of the series data and Black line represents standard deviation of the series. ",
                "dataframe" : stationarity['dataframe'],
                "rollmean" : stationarity['rollmean'],
                "rollstd" : stationarity['rollstd'],
                # "stationarity_check": json.dumps(stationarity, default=default)
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500


@app.get('/stationaritytest')
def stationarity_test():

    target_col = cache.get('target_col')
    df = cache.get('df1')

    if (target_col):
        try:
            stationarity = timeseries.stationarity_test(df, df[target_col])
            print(stationarity.keys())
            if 'final_msg' in stationarity.keys():
                return jsonify({
                    "message": "Success1",
                    "adf_test": stationarity['adf_test'],
                    "kpss_test" : stationarity['kpss_test'],
                    "final_msg" : stationarity['final_msg'],
                }), 200
            else:
                return jsonify({
                    "message": "Success2",
                    "adf_test": stationarity['adf_test'],
                    "kpss_test" : stationarity['kpss_test'],
                    "conversion_dict" : stationarity['conversion_dict'],
                    "final_test" : stationarity['final_test'],
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

@app.get('/tsplot')
def acf_and_pacf():
    target_col = request.headers['target_col']
    nlags = int(request.headers['nlags'])

    if (target_col):
        try:
            df = cache.get('df1')
            acf_fig = timeseries.create_corr_plot(df[target_col], False, nlags)
            pacf_fig = timeseries.create_corr_plot(df[target_col], True, nlags)

            acf_div = pyo.plot(acf_fig, output_type='div', include_plotlyjs=False)#acf_fig.to_html() 
            pacf_div = pyo.plot(pacf_fig, output_type='div', include_plotlyjs=False) #pacf_fig.to_html()
            
            return jsonify({
                "message": "Success",
                "acf_div":  acf_div, #json.dumps(acf_div, default=default),
                "pacf_div": pacf_div, #json.dumps(pacf_div, default=default)
            }), 200

        except Exception as ex:
            print(ex)
            return jsonify({"message": "Something went wrong."}), 500


# @app.get('/autocorrelation')
# def acf_and_pacf():
#     if (request.headers['tar_col']):
#         tar_col = request.headers['tar_col']
#         try:
#             graph = timeseries.acf_pacf(final_data[tar_col])

#             return jsonify({
#                 "message": "Success",
#                 "acf and pacf": json.dumps(graph, default=default)
#             }), 200

#         except Exception as ex:
#             print(ex)
#             return jsonify({"message": "Something went wrong."}), 500
