# Base-code
These are code bases that are helpful in research such as ML gradient ascent models and Stats



import dash
from dash import html, dash_table, Input, Output, State, ctx

import pandas as pd

app = dash.Dash(__name__)

# Sample DataFrame
df = pd.DataFrame({
    'id': [0, 1, 2],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 30, 22],
})

# App Layout
app.layout = html.Div([
    html.Div([
        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[
                    {'name': 'Name', 'id': 'Name'},
                    {'name': 'Age', 'id': 'Age'},
                ],
                data=df.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_table={'display': 'inline-block'},
            ),
        ], style={'width': '60%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                html.Button('Click Me', id={'type': 'row-button', 'index': i})
                for i in df['id']
            ])
        ], style={'display': 'inline-block', 'paddingLeft': '20px'}),
    ]),
    
    html.Div(id='output', style={'marginTop': '20px'})
])

# Callback to detect which button was clicked
@app.callback(
    Output('output', 'children'),
    Input({'type': 'row-button', 'index': dash.ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def handle_row_button_click(n_clicks):
    triggered = ctx.triggered_id
    if triggered is not None:
        return f"You clicked the button for row with id: {triggered['index']}"
    return "No button clicked."

if __name__ == '__main__':
    app.run_server(debug=True)

