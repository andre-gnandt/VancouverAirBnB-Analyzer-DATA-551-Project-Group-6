import dash
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div('I am alive!!')
app.run_server()