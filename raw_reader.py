import numpy as np
import plotly.graph_objects as go
import os

print('Loading data...')
file = os.listdir('case/postProcessing/wallPressure')[-1]
print(f'file: {file}')
data = np.loadtxt(f'case/postProcessing/wallPressure/{file}/p_airfoil.raw', dtype=np.float32)
print(data.shape)

print('removing useless sections...')
data = data[data[:, 2] == 0.]
print(data.shape)



fig = go.Figure(data=[go.Scatter3d(
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=data[:, 3],  # set color to an array/list of desired values
        colorscale='Turbo',  # choose a colorscale
        opacity=0.8,
        colorbar=dict(thickness=20)
    )
)],
    layout=go.Layout(
        width=1000,
        height=600,
    )
)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), )
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=0.1, z=1))
# fig.update_zaxes(range=[0, 1])
fig.show()
