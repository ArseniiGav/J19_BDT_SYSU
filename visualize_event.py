import numpy as np
import pandas as pd
from _plotly_future_ import v4_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly

import plotly.graph_objs as go
from plotly.subplots import make_subplots

init_notebook_mode(connected=False)
import plotly.io as pio
pio.templates.default = 'plotly_dark'

def visualize_events(evtIDs, R, lpmt_x_array, lpmt_y_array, lpmt_z_array, lpmt_s_array, event_pos_x_array, event_pos_y_array, event_pos_z_array, event_edep_array):
    theta = np.linspace(0,2*np.pi,100)
    phi = np.linspace(0,np.pi,100)
    x = np.outer(np.cos(theta),np.sin(phi))
    y = np.outer(np.sin(theta),np.sin(phi))
    z = np.outer(np.ones(100),np.cos(phi))

    fig = go.Figure()

    for i in range(len(evtIDs)):
      fig.add_trace(go.Scatter3d(x=lpmt_x_array[i], y=lpmt_y_array[i], z=lpmt_z_array[i],
                                 mode='markers',
                                 visible=(i == 0),
                                 text=lpmt_s_array[i],
                                 marker=dict(size=2.75,
                                             color=lpmt_s_array[i],
                                             colorscale=['rgb(150, 30, 70)', 'rgb(255, 200, 70)', 'rgb(255, 255, 100)'],
                                             opacity=1),
                                 name='LPMT'))

    for i in range(len(evtIDs)):
      fig.add_trace(go.Scatter3d(
        x=np.array(event_pos_x_array[i]),
        y=np.array(event_pos_y_array[i]),
        z=np.array(event_pos_z_array[i]),
        visible=(i == 0),
        mode='markers',
        text='Edep ' + str(event_edep_array[i])[:5],
        marker=dict(size=7,
                    color='white',
                    # colorscale='portland',
                    opacity=1),
        name='Event. ' + 'Edep ' + str(event_edep_array[i])[:5]))

    for i in range(len(evtIDs)):
      fig.add_trace(go.Surface(
        x=x * R,
        y=y * R,
        z=z * R,
        opacity=0.3,
        visible=(i == 0),
        showscale=False,
        colorscale=['rgb(2, 2, 2)', 'rgb(4, 4, 4)'],
        name=''
      )
      )

    buttons = []
    for N in range(0, len(evtIDs)):
      buttons.append(
        dict(
          args=['visible', [False] * N + [True] + [False] * (len(evtIDs) - 1 - N)],
          label='EvtID = {}'.format(evtIDs[N]),
          method='restyle'
        )
      )

    fig.update_layout(
      updatemenus=list([
        dict(
          x=-0.05,
          y=1,
          yanchor='top',
          buttons=buttons
        ),
      ]),
      scene_camera_eye=dict(x=1, y=1, z=1)

    )


    fig.show()

