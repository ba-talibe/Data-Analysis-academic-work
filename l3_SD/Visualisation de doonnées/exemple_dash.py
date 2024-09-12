from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

app = Dash(__name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

app.title = "Dashboard"
server = app.server
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


df = px.data.gapminder()
countries = df['country'].unique()
years = df['year'].unique()
indicators = ['lifeExp','gdpPercap','pop']

app.layout = html.Div([
    html.Div([
        html.H4('Visualisation de l\'espérance de vie ', className="app__header__title"),
        html.Div([
            dcc.Dropdown(
                id="country",
                options=countries,
                value=countries[0],
                clearable=False,
                className="link-button-3")
        ]),
        dcc.Graph(id="lifeExpGraph")
    ]),
    html.Div([
        html.H4('Visualisation "Libre" ', className="app__header__title"),
        html.Div([
            dcc.Dropdown(
                id="x",
                options=indicators,
                value=indicators[0],
                clearable=False,
                style={"width": "200px","display":"flex"},
                className="link-button-3"),
            dcc.Dropdown(
                id="y",
                options=indicators,
                value=indicators[1],
                clearable=False,
                style={"width": "200px","display":"flex"},
                className="link-button-3"),
            dcc.Dropdown(
                id="size",
                options=indicators,
                value=indicators[2],
                clearable=False,
                style={"width": "200px","display":"flex"},
                className="link-button-3"),
            dcc.Dropdown(
                id="year",
                options=years,
                value=years[0],
                clearable=False,
                style={"width": "200px","display":"flex"},
                className="link-button-3"),
            dcc.Checklist(['xlog','ylog'],inline=True,id="check")
        ]),
        dcc.Graph(id="freeGraph")        
    ])
])


@app.callback(
    Output("lifeExpGraph", "figure"), 
    Input("country", "value")
)
def display_lifeExp(country):
    data = df[df['country']==country]
    fig = px.line(data,x='year',y='lifeExp')
    return fig

@app.callback(
    Output("freeGraph","figure"),
    Input("x","value"),
    Input("y","value"),
    Input("size","value"),
    Input("year","value"),
    Input("check","value")
)
def display_freeGraph(x,y,size,year,check):
    data = df[df['year']==year]
    print(check)
    fig = px.scatter(data,x=x,y=y,size=size,color="continent",hover_name="country",size_max=60)
    return fig

if __name__ == "__main__":
    app.run(jupyter_mode="jupyterlab",port=8081)

