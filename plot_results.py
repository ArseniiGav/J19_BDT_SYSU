from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=False)
import plotly.io as pio
pio.templates.default = 'plotly_white'

from neptune.new.types import File


def plot_model_comparison(run, dfs, run_name, colors, names, shifts,
                          ylim=3, ratios_lims=[0.95, 1.05],
                          ratio_ticks=[0.95, 1.0, 1.05, 1.1], base_ind=3, 
                          opacity=0.8):
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.005,
                        row_width=[0.25, 0.25, 0.5]
    )

    for k in range(len(dfs)):
        fig.add_trace(go.Scatter(
                        x=dfs[k].energy+1.022+shifts[k],
                        y=100.*dfs[k].res,
                        mode='markers',
                        marker=dict(
                                color=colors[k],
                                symbol=k,
                                opacity=opacity
                            ),
                        error_y=dict(
                                type='data',
                                array=100.*dfs[k].res_err,
                                visible=True,
                                width=8,
                            ),
                        name=names[k],
                        showlegend=True,
            ), row=1, col=1)

    for k in range(len(dfs)):
        if k != base_ind:
            fig.add_trace(go.Scatter(
                            x=dfs[k].energy+1.022+shifts[k],
                            y=dfs[k]['res'].to_numpy()/dfs[base_ind]['res'].to_numpy(),
                            mode='markers',
                            marker=dict(
                                    color=colors[k],
                                    symbol=k,
                                    opacity=opacity
                                ),
                            error_y=dict(
                                    type='data',
                                    array=dfs[k]['res']*100.*( (dfs[k]['res_err'].to_numpy()/dfs[k]['res'].to_numpy())**2 + 
                                                       (dfs[base_ind]['res_err'].to_numpy()/dfs[base_ind]['res'].to_numpy())**2 )**0.5,
                                    visible=True,
                                    width=8,
                                ),
                            name=names[k],
                            showlegend=False,

                ), row=2, col=1)

    fig.add_hrect(y0=1, y1=1, line_width=1.5, line_color=colors[base_ind], line_dash="dashdot", row=2, col=1)

    for k in range(len(dfs)):
        fig.add_trace(go.Scatter(
                        x=dfs[k].energy+1.022+shifts[k],
                        y=100.*dfs[k].bias,
                        mode='markers',
                        marker=dict(
                                color=colors[k],
                                symbol=k,
                                opacity=opacity
                            ),
                        error_y=dict(
                                type='data',
                                array=100.*dfs[k].bias_err,
                                visible=True,
                                width=8
                            ),
                        name=names[k],
                        showlegend=False,
            ), row=3, col=1)

    fig.update_layout(

        xaxis3 = dict(
            showline=True,
            title_text="Visible energy, MeV",
            ticks='outside',
            mirror=True,
            linecolor='black',
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        xaxis2 = dict(
            showline=True,
            ticks='outside',
            mirror=True,
            tick0=1,
            dtick=1,
            linecolor='black',
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        xaxis1 = dict(
            showline=True,
            ticks='outside',
            mirror=True,
            linecolor='black',
            showgrid=True,
            tick0=1,
            dtick=1,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        yaxis3 = dict(
            showline=True,
            ticks='outside',
            title_text="Bias, %",
            mirror=True,
            linecolor='black',
            range=[-0.5, 0.5],
            tickmode='array',
            tickvals=[-0.25, 0, 0.25],
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=0.25
        ),

        yaxis2 = dict(
            showline=True,
            ticks='outside',
            title_text="Ratios",
            mirror=True,
            linecolor='black',
            range=ratios_lims,
            tickmode='array',
            tickvals=ratio_ticks,
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        yaxis1 = dict(
            showline=True,
            ticks='outside',
            title_text="Resolution, %",
            mirror=True,
            linecolor='black',
            range=[0.95, ylim],
            tickmode='linear',
            tick0=1,
            dtick=0.5,
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        font=dict(
            size=15,
        )

    )

    fig.update_layout(
        legend=dict(
            x=0.72,
            y=0.98,
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=14,
                color="black"
            ),
    #         bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )


    fig.show()
    pio.write_image(fig, '{}/results/models_comparison.pdf'.format(run_name), width=600, height=600, scale=1)
    run['output/models_comparison'].upload(File.as_html(fig))
    
def plot_res_and_bias(run, df_bdt, run_name, file_name, mode,
                      symbols, colors, options, names,
                      ylim=3, errors=False, opt=0,
                      bias_lims=[-0.35, 0.35], legend_x=0.55,
                      legend_y=0.98, opacity=0.7):
    
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.005,
                        row_width=[0.25, 0.75]
    )
    
    if mode=='options':
        df = lambda k: df_bdt[(df_bdt['opt']==options[k]) & (df_bdt['model']=='5M')]
    if mode=='datasets':
        df = lambda k: df_bdt[(df_bdt['opt']==23) & (df_bdt['model']==options[k])]
    if mode=='opt_vs_basic':
        df = lambda k: df_bdt[(df_bdt['opt']==opt) & (df_bdt['model']==options[k])]
    
    for k in range(len(options)):
        if errors:
            error_y=dict(
                        type='data',
                        array=100.*df(k).res_err,
                        visible=True,
                        width=10
            )
        else:
            error_y=dict()
        fig.add_trace(go.Scatter(
                        x=df(k).energy+1.022,
                        y=100.*df(k).res,
                        mode='markers',
                        marker=dict(
                                symbol=symbols[k],
                                color=colors[k],
                                size=8,
                                opacity=opacity
                            ),
                        error_y=error_y,
                        name=names[k],
                        showlegend=True,
            ), row=1, col=1)

    for k in range(len(options)):
        if errors:
            error_y=dict(
                        type='data',
                        array=100.*df(k).bias_err,
                        visible=True,
                        width=10
            )
        else:
            error_y=dict()   
        fig.add_trace(go.Scatter(
                        x=df(k).energy+1.022,
                        y=100.*df(k).bias,
                        mode='markers',
                        marker=dict(
                                symbol=symbols[k],
                                color=colors[k],
                                size=8,
                                opacity=opacity
                            ),
                        error_y=error_y,
                        name=names[k],
                        showlegend=False,
            ), row=2, col=1)

    fig.update_layout(

        xaxis2 = dict(
            title_text="Visible energy, MeV",
            showline=True,
            ticks='outside',
            mirror=True,
            tick0=1,
            dtick=1,
            linecolor='black',
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        xaxis1 = dict(
            showline=True,
            ticks='outside',
            mirror=True,
            linecolor='black',
            showgrid=True,
            tick0=1,
            dtick=1,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        yaxis2 = dict(
            showline=True,
            ticks='outside',
            title_text="Bias, %",
            mirror=True,
            linecolor='black',
            range=bias_lims,
            tickmode='array',
            tickvals=[-0.25, 0, 0.25],
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=0.25
        ),

        yaxis1 = dict(
            showline=True,
            ticks='outside',
            title_text="Resolution, %",
            mirror=True,
            linecolor='black',
            range=[0.95, ylim],
            tickmode='linear',
            tick0=1,
            dtick=0.5,
            showgrid=True,
            gridcolor='grey',
            gridwidth=0.25,
        ),

        font=dict(
            size=15,
        )

    )

    fig.update_layout(
        legend=dict(
            x=legend_x,
            y=legend_y,
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=14,
                color="black"
            ),
    #         bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )

    #diff_options_datasets_errors
    fig.show()
    pio.write_image(fig, '{}/results/{}.pdf'.format(run_name, file_name), width=600, height=500, scale=1)
    run['output/{}'.format(file_name)].upload(File.as_html(fig))