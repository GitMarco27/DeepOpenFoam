import numpy as np
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

import plotly.express as px
# from stable_baselines3.common.utils import set_random_seed


import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
# from stable_baselines3.common.monitor import Monitor
import dash_bootstrap_components as dbc

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
# data una figura aggiunge un grafico ad essa
def add_trace(fig, data, name=''):
    x = data[:, 0]
    y = data[:, 1]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    return fig


def get_slider(name_slider, index, min_value=0, max_value=20, init_value=4, is_vertical: bool = False,
               step_value: float = 0.5):
    return html.Div([
        html.Div([
            html.H5(index)], style={'marginLeft': '30%'}),
        html.Div([
            dcc.Slider(
                id=name_slider,
                min=min_value,
                max=max_value,
                value=init_value,
                tooltip={"placement": "bottom", "always_visible": True},
                # marks={i: '{}'.format(10 ** i) for i in range(4)},
                step=step_value,
                vertical=is_vertical
            )
        ])
    ]
    )


def get_starting_geom_slider(name_slider, index, min_value=0, max_value=20, init_value=4, is_vertical: bool = False,
                             step_value: float = 0.5, eff=0, dim_slider: int = 40):
    return html.Div([
        html.Div([
            html.H5(index)]),

        html.Div([
            dcc.Slider(
                id=name_slider,
                min=min_value,
                max=max_value,
                value=init_value,
                tooltip={"placement": "bottom", "always_visible": True},
                # marks={i: '{}'.format(10 ** i) for i in range(4)},
                step=step_value,
                vertical=is_vertical
            )
        ]),

        html.Div(
            children=f"Starting efficiency: {eff}", id="starting_eff",
            style={'marginLeft': '0px', 'marginTop': '10px'}),
        html.Div(
            children=f"Efficiency obtained: {eff}", id="obtained_eff", style={'marginLeft': '0px', }),
    ])

def create_dash(env):
    app = JupyterDash(__name__)
    slider_index = 0
    min_latent = -4
    max_latent = 4
    list_slider = np.array([0, 1, 2, 3, 4])

    input_slider = [Input(f'slider_{i}', 'value') for i in range(5)]
    output_slider = [Output(f'slider_{i}', 'value') for i in range(5)]

    sidebar = html.Div(
        [
            html.H2("Sidebar", className="display-4"),
            html.Hr(),
            dbc.Nav(
                [

                    get_starting_geom_slider(f'slice_geometry_start',
                                             'Select a starting geometry and click the button to update the dash',
                                             min_value=0, max_value=len(env.data_env['cod']),
                                             init_value=0, is_vertical=False, step_value=1, eff=starting_eff,
                                             dim_slider=80),

                    html.H5('Latent Params', className='text-center'),

                    html.Div([get_slider(f'slider_{index_latent}', 'l' + str(index_latent), min_value=int(min_latent),
                                         max_value=int(max_latent), step_value=0.1,
                                         init_value=current_latent[index_latent], is_vertical=False) for index_latent in
                              list_slider
                              ]),

                    html.Div([
                        html.Button('<--', id='prev_b', n_clicks=0, style={'marginLeft': '10%', 'marginTop': '10%'}),
                        html.Button('-->', id='next_b', n_clicks=0, style={'marginLeft': '60%', 'marginTop': '10%'}),
                    ]),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div([
        # html.Div([dcc.Graph(id="graph-picture", figure=fig_image),
        #             ], style={'marginLeft':'32%'}),
        html.Div([
            dcc.Graph(id='geom_figure'),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='figure_fields'),
        ], style={'display': 'inline-block', 'width': '40%'}),
    ], style=CONTENT_STYLE, )

    app.layout = html.Div([
        dcc.Location(id="url"),
        sidebar,
        content
    ])

    # @app.callback(
    #     Output('container-button-timestamp', 'children'),
    #     Input('prev_b', 'n_clicks'),
    #     Input('next_b', 'n_clicks'),
    # )
    # def displayClick(btn1, btn2, btn3):
    #     changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    #     if 'btn-nclicks-1' in changed_id:
    #         msg = 'Button 1 was most recently clicked'
    #     elif 'btn-nclicks-2' in changed_id:
    #         msg = 'Button 2 was most recently clicked'
    #     elif 'btn-nclicks-3' in changed_id:
    #         msg = 'Button 3 was most recently clicked'
    #     else:
    #         msg = 'None of the buttons have been clicked yet'
    #     return html.Div(msg)

    @app.callback(
        Output("starting_eff", "children"),
        output_slider,
        Input('slice_geometry_start', 'value'),
        Input('starting_eff', "children"),
        Input('prev_b', 'n_clicks'),
        Input('next_b', 'n_clicks'),
        input_slider
    )
    def update_starting_geometry(slice_geometry_start, starting_eff, *sliders):
        global current_latent
        global list_slider

        ctx = dash.callback_context
        # trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        changed_id = [p['prop_id'] for p in ctx.triggered][0]

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        sliders = list(sliders[:5])

        if "slice_geometry_start" in changed_id:
            current_latent = env.data_env['cod'][slice_geometry_start]
            gv = env.get_global_variable_from_state()
            starting_eff = f"Starting efficiency:{gv[0] / gv[1]}"
            for index, value in enumerate(list_slider):
                sliders[index] = current_latent[value]
        elif 'next_b' in changed_id:
            if list_slider[-1] + 5 < current_latent.shape[0] - 1:
                list_slider = list_slider + 5
                for index, value in enumerate(list_slider):
                    sliders[index] = current_latent[value]
        elif 'prev_b' in changed_id:
            if list_slider[0] - 5 >= 0:
                list_slider = list_slider - 5
                for index, value in enumerate(list_slider):
                    sliders[index] = current_latent[value]
        return starting_eff, *sliders

    @app.callback(
        Output('geom_figure', 'figure'),
        Output('figure_fields', 'figure'),
        Output("obtained_eff", "children"),
        Input('slice_geometry_start', 'value'),
        input_slider)
    def update_figure(slice_geometry_start, *sliders):
        global current_latent
        # trasformo i valori degli slider in un numpy cosi da ricreare i latent

        for index, value in enumerate(list_slider):
            current_latent[value] = sliders[index]

        current_geom = env.decode()
        global_variables = env.pred_global_variables()
        eff = global_variables[0] / global_variables[1]

        # figura 1
        fig_1 = go.Figure(layout=go.Layout(
            width=800,
            height=600,
        ))
        add_trace(fig_1, current_geom[0], name=f'X_or')

        fig_1.update_layout(title={
            'text': "New Geom",
            'x': 0.5}, yaxis_range=[-0.52, 0.51], xaxis_range=[-0.02, 1.01],
        )
        fig_1.layout.xaxis.fixedrange = True
        fig_1.layout.yaxis.fixedrange = True

        # figura 2
        #     field_geom = X_train[slice_geometry_start].reshape(1,-1,3)
        #     temp_pointcloud = Pointcloud(current_geom[0][:,0], current_geom[0][:,1], current_geom[0][:,2])
        #     pred_temperature, pred_pressure = calc_fields(field_geom, model_reg_fields, loader)

        #     if field_type == 'Temperature':
        #         fig_2 = temp_pointcloud.plot_geometry(color=pred_temperature, size=(600, 600), title= {'text': "Temperature field",'x':0.5})
        #     else:
        #         fig_2 = temp_pointcloud.plot_geometry(color=pred_pressure, size=(600, 600),  title={'text': 'Pressure Field','x':0.5} )

        #     # fig_2.layout.scene.camera.eye = fig_1.layout.scene.camera.eye
        fig_2 = fig_1

        return fig_1, fig_2, f"Efficiency obtained: {eff}"

    app.run_server(mode='external', port=6006, host='0.0.0.0')
