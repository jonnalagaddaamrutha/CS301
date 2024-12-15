from dash import Dash, dcc, html, Input, Output, State, callback
import pandas as pd
import base64
import io
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Dash(__name__)

global_model = None

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([html.A('Upload File')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderRadius': '5px',
            'textAlign': 'center',
            'backgroundColor': '#f0f0f0',
            'fontSize': '16px',
        },
    ),
    dcc.Store(id='stored-data'),

    html.Div([
        html.Div(children=[
            html.Div("Select Target:", style={'width': '6%', 'lineHeight': '10px', 'height': '10px'}),
            dcc.Dropdown(
                id='target-dropdown',
                options=[],
                value=None,
                placeholder='Select a target variable',
                style={'width': '40%', 'height': '40px', 'lineHeight': '10px', 'textAlign': 'left'}
            ),
        ], style={'width': '100%', 'height': '60px', 'backgroundColor': '#f0f0f0', 'marginTop': '10px',
                  'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),

    ], style={'display': 'center', 'width': '100%'}),

    html.Div([
        dcc.RadioItems(id='categorical-variable-radio', options=[], value=None)
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'left', 'marginTop': '20px'}),

    html.Div([
        html.Div(id='avg-target-bar-chart', style={'width': '48%', 'display': 'inline-block'}),
        html.Div(id='correlation-bar-chart', style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    html.Div([
        dcc.Checklist(id='feature-checklist', options=[], value=[], inline=True),
        html.Button('Train Model', id='train-button', n_clicks=0),
        html.Div(id='r2-score-output', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginTop': '20px', 'textAlign': 'center'}),


    html.Div([
        dcc.Input(
            id='feature-input',
            type='text',
            placeholder='Enter feature values, comma-separated',
            style={'width': '60%', 'padding': '10px', 'fontSize': '14px'}
        ),
        html.Button('Predict', id='predict-button', n_clicks=0, style={'marginLeft': '10px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),

    html.Div(id='predicted-output', style={'marginTop': '20px', 'fontSize': '16px', 'textAlign': 'center'}),

    dcc.Store(id='trained-model', data=None),

    html.Div(id='output-data')
])

def process_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    cat_cols = df.select_dtypes(include=["object"]).columns

    original_cat_cols = cat_cols.tolist()

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df_encoded, original_cat_cols

@callback(
    [Output('output-data', 'children'),
     Output('stored-data', 'data'),
     Output('target-dropdown', 'options'),
     Output('categorical-variable-radio', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def file(contents, filename):
    if contents is None:
        return html.Div([]), None, [], []

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div(['Unsupported file format. Please upload a CSV file.']), None, [], []

        processed_df, original_cat_cols = process_data(df)

        numerical_cols = processed_df.select_dtypes(include=['number']).columns
        target_options = [{'label': col, 'value': col} for col in numerical_cols]

        cat_options = [{'label': col, 'value': col} for col in original_cat_cols]

        return html.Div([]), processed_df.to_dict('records'), target_options, cat_options

    except Exception as e:
        return html.Div([f'There was an error processing this file: {str(e)}']), None, [], []

@callback(
    Output('avg-target-bar-chart', 'children'),
    [Input('target-dropdown', 'value'), Input('categorical-variable-radio', 'value')],
    [State('stored-data', 'data')]
)
def update_avg_bar_chart(target, cat_var, data):
    if not target or not cat_var or not data:
        return html.Div(['Please select both a target and a categorical variable.'])

    df = pd.DataFrame(data)

    encoded_cat_columns = [col for col in df.columns if col.startswith(cat_var)]

    if not encoded_cat_columns:
        return html.Div([f"No one-hot encoded columns found for '{cat_var}'"])

    avg_data = df.groupby(encoded_cat_columns)[target].mean().reset_index()
    fig = px.bar(avg_data, x=encoded_cat_columns, y=target, title=f'Average {target} by {cat_var}')

    fig.update_layout(
        xaxis_title=f'{cat_var}',
        yaxis_title=f'{target} (average)'
    )

    return dcc.Graph(figure=fig)

@callback(
    Output('correlation-bar-chart', 'children'),
    [Input('target-dropdown', 'value')],
    [State('stored-data', 'data')]
)
def update_corr_bar_chart(target, data):
    if not target or not data:
        return html.Div(['Please select a target variable.'])

    df = pd.DataFrame(data)

    numeric_df = df.select_dtypes(include=['number'])

    if target not in numeric_df.columns:
        return html.Div(['Selected target variable is not numeric or not found in the dataset.'])

    corr_data = numeric_df.corr()[target].abs().sort_values(ascending=False).reset_index()
    corr_data.columns = ['Variable', 'Correlation']

    corr_data = corr_data[corr_data['Variable'] != target]

    fig = px.bar(corr_data, x='Variable', y='Correlation', title=f'Correlation of Numerical Variables with {target}')

    fig.update_layout(
        xaxis_title='Numerical Variables',
        yaxis_title='Correlation Strenth (abslute value)'
    )

    return dcc.Graph(figure=fig)

@callback(
    Output('feature-checklist', 'options'),
    [Input('stored-data', 'data')]
)
def update_feature_checklist(data):
    if not data:
        return []
    df = pd.DataFrame(data)
    feature_options = [{'label': col, 'value': col} for col in df.columns]
    return feature_options

@callback(
    Output('r2-score-output', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('stored-data', 'data'),
     State('feature-checklist', 'value'),
     State('target-dropdown', 'value')]
)

def train_model(n_clicks, data, selected_features, target):
    global global_model

    if n_clicks == 0 or not data or not selected_features or not target:
        return "Please upload data, select features, and choose a target variable before training.", None

    df = pd.DataFrame(data)
    X = df[selected_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X.select_dtypes(include=['number']).columns
    cat_features = X.select_dtypes(include=['object']).columns

    num_transformer = SimpleImputer(strategy='mean')
    cat_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    global_model = model

    return f"The R^2 Score is: {r2:.2f}"

@callback(
    Output('predicted-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('feature-input', 'value'),
     State('stored-data', 'data'),
     State('feature-checklist', 'value')]
)
def predict_output(n_clicks, input_values, data, selected_features):
    global global_model

    if n_clicks > 0:
        if input_values:
            try:
                value, column_name = input_values.split(',')

                value = float(value.strip())
                column_name = column_name.strip()

                if column_name not in selected_features:
                    return f"Error: Column '{column_name}' is not in the selected features."

                input_data = {col: [0] for col in selected_features}
                input_data[column_name] = [value]
                input_df = pd.DataFrame(input_data)

                if global_model is None:
                    return "Error: The model is not trained yet."

                # Predict using the model
                prediction = global_model.predict(input_df)
                return f"Predicted value: {prediction[0]:.2f}"
            except ValueError:
                return "Error: Please provide a valid numeric value and column name (e.g., '32,age')."
            except Exception as e:
                return f"Error processing input: {str(e)}"
        else:
            return "Please enter the feature value and column name (e.g., '32,age')."
    return ""

if __name__ == '__main__':
    app.run(debug=True)