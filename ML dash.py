import base64
#import datetime
import io
import plotly.graph_objs as go
import cufflinks as cf
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
"graphBackground": "#F5F5F5",
"background": "#ffffff",
"text": "#000000"
}

app.layout = html.Div([
    html.Div([
        html.H3(' DASHBOARD! ',style=dict(font=dict(family='times new roman'),color='#3BB9FF'
                                          ,width='50%',display='inline-block',textAlign='center',
                                          border={'color':'red',
                                                                    'border':'5px solid'}))
    ]),
dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'width': '97%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'alighn' : 'center'
    },
    # Allow multiple files to be uploaded
    multiple=True
),
    html.Div([
        html.H4('Chosse graph for data visulization: ',style=dict(width='40%',display='inline-block',padding='10px')),
        dcc.Dropdown(id = 'Plot_type', options=[{'label': 'Scatter' ,'value': 'scatter' },
                                                {'label':'Stack_BAR','value':'StackBar'},
                                                {'label':'Bar','value':'Bar'}],
                 value='scatter',style=dict(display='inline-block',width='55%'))]),
   html.Div([
        html.H5("Select X axis",style=dict()),
        dcc.Dropdown(id='x_axis')
    ], style={'width': '40%', 'display': 'inline-block','padding':10}),
    html.Div([
        html.H5("Select Y axis", style=dict()),
        dcc.Dropdown(id='y_axis')
    ], style={'width': '40%', 'display': 'inline-block','padding':10}),

dcc.Graph(id='Mygraph'),
html.Div(id='output-data-upload'),
    html.Div([
      html.H3('MACHIENE LEARNING MODULE',style=dict(display='inline-block',width='50%')),
                dcc.Dropdown(id = 'ML_type', options=[{'label': 'Linear Regression' ,'value': 'Linear regression' },
                                                {'label':'Polynomial Regression','value':'Polynomial_r'},
                                                {'label':'Logistic','value':'Logistic_R'}],
                 value='linear regression',style=dict(display='inline-block',width='50%')),
            html.Div(children=[
                    html.H3('Prediction Accuracy '),

                    html.H5(id='prediction'),
                    html.Br()
                    ], style=dict(display='inline-block', width='30%',height='150px',overflow = 'auto')),
            dcc.Graph(id='lineGraph',style=dict(display='inline-block',width='70%'))
    ])

])
@app.callback(Output('x_axis', 'options'),
              [ Input('upload-data', 'contents'),
                Input('upload-data', 'filename'),]
              )
def Update_xaxis(contents,filename):
    feature = ['none']
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        #df = df.set_index(df.columns[0])
        feature = df.columns
    return [{'label': i, 'value': i} for i in feature]

@app.callback(Output('y_axis', 'options'),
              [ Input('upload-data', 'contents'),
                Input('upload-data', 'filename'),]
              )
def Update_xaxis(contents,filename):
    feature = ['none']
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        #df = df.set_index(df.columns[0])
        feature = df.columns
    return [{'label': i, 'value': i} for i in feature]


@app.callback([Output('lineGraph', 'figure'),
Output('prediction', 'children')],[
Input('upload-data', 'contents'),
Input('upload-data', 'filename'),
Input('x_axis','value'),
Input('y_axis', 'value'),
Input('ML_type', 'value')])
def ML_mod(contents,filename,X_N,Y_N,ML_select):
    x = []
    y = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        # df = df.set_index(df.columns[0])
        print(df)
        feature = df.columns
        if (X_N != None and Y_N != None ):
            x = df[X_N]
            y = df[Y_N]
        else:
            x=df[feature[0]]
            y=df[feature[0]]
    predict_y = []
    a = np.array(x)
    b = np.array(y)
    if (ML_select=="Polynomial_r"):
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.2, random_state=0)

            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=4)
            X_poly = poly.fit_transform(a)

            poly.fit(X_poly, b)
            predict_y = poly.predict(X_test)
        except:
            print(predict_y)
    else:
        try:
            am = sum(a)/len(a)
            bm = sum(b)/len(b)
            avg = 0
            sq = 0
            for  i in range(len(a)):
                avg = avg +(a[i]-am)*(b[i]-bm)
                sq = sq + (a[i]-am)*(a[i]-am)
            slope = avg/sq
            intercept = bm-slope*am

            for i in a:
                predict_y.append(intercept+ slope*i)
        except:
            print(predict_y)
        '''
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.2, random_state=0)
    
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(a)

    poly.fit(X_poly, b)
    predicted = lr.predict(X_test)
    
    '''
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker = {
                         'color': 'rgb(255,0,0)',
                         'line': {'width': 1}
                     }),

            go.Scatter(
                x=x,
                y=predict_y,
                mode='lines+markers',
                marker={
                    'color': 'rgb(0,255,0)',
                    'line': {'width': 1}
                },
            name='Predicted Y'),

        ],
        layout=go.Layout(
            plot_bgcolor='#000000',
            paper_bgcolor='#000000'
        ))


    return fig,'Outpult : {}'.format(predict_y)

@app.callback(Output('Mygraph', 'figure'), [
Input('upload-data', 'contents'),
Input('upload-data', 'filename'),
Input('Plot_type', 'value'),
Input('x_axis','value'),
Input('y_axis', 'value')
])
def update_graph(contents, filename,yy,X_N,Y_N):
    x = []
    y = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        #df = df.set_index(df.columns[0])
        #print(df)
        feature = df.columns
        #print(feature)
        if (X_N != None and Y_N != None ):
            x = df[X_N]
            y = df[Y_N]
        else:
            x=df[feature[0]]
            y=df[feature[0]]
    if yy=='scatter':
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers')
                ],
            layout=go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"]
            ))
        return fig
    elif yy == 'StackBar':
        fig = go.Figure(
            data = [
                go.Bar(
                x = df.index,
                y = df[response],
                name=response
                ) for response in df.columns],
            layout=go.Layout(title='Barchart',barmode='stack',plot_bgcolor=colors["graphBackground"],
                                paper_bgcolor=colors["graphBackground"]
                             #font={'color': colors['text']}
                                     )   )
        return fig
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df.index,
                    y=df[response],
                    name=response
                ) for response in df.columns],
            layout=go.Layout(title='Barchart',plot_bgcolor=colors["graphBackground"],
                             paper_bgcolor=colors["graphBackground"]
                             # font={'color': colors['text']}
                             ))
        return fig


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


@app.callback(Output('output-data-upload', 'children'),
          [
Input('upload-data', 'contents'),
Input('upload-data', 'filename')
])
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div([
            html.H5(filename),
            dash_table.DataTable(
                data=df.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_action = 'none',
                style_table = {'height': '300px', 'overflowY': 'auto'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },
            ),

        ])

    return table


if __name__ == '__main__':
    app.run_server(debug=True)