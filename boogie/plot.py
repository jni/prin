"""
Compute pagerank on a bunch of datasets and plot it against the in-degree.

The point is to find points that have high PR but relatively low in-degree
and hold them up as examples of PR success.

Based on experience these are hard to come by. =P
"""
import os
import sys
import argparse

import numpy as np
import feather

from scipy import sparse
from matplotlib import cm
import networkx as nx

import pandas as pd

from bokeh.models import (BoxSelectTool, Select, PanTool,
                          ResizeTool, ResetTool,
                          HoverTool, WheelZoomTool)
TOOLS = [BoxSelectTool, PanTool, WheelZoomTool, ResizeTool, ResetTool]
from bokeh.models import ColumnDataSource
from bokeh import plotting as bplot
from bokeh import layouts
#from bokeh.plotting import figure, gridplot, output_file, show

from .spectral import compute_pagerank, node_coordinates


def network_properties(network : nx.DiGraph,
                       in_degree_threshold : float = -1,
                       pagerank_threshold : float = -1,
                       damping : float = 0.85,
                       spectral_offset : float = 0.5)\
        -> (pd.DataFrame, sparse.spmatrix):
    conn = max(nx.connected_components(network.to_undirected()), key=len)
    conn = nx.subgraph(network, conn)
    pr = compute_pagerank(conn, damping=damping)
    names = nx.nodes(conn)
    indeg = [conn.in_degree(n) for n in names]
    odeg = [conn.out_degree(n) for n in names]
    description = [conn.node[n].get('description', n) for n in names]
    x, y, z, Adj, aff_names = node_coordinates(conn, nodelist=names,
                                               offset=spectral_offset)
    data = {'id': names,
            'in_degree': indeg,
            'out_degree': odeg,
            'pagerank': pr,
            'affinity_x': x,
            'affinity_y': y,
            'processing_depth': z,
            'description': description}
    df = pd.DataFrame(data, index=names)
    df = df[df['pagerank'] > pagerank_threshold / len(names)]
    df = df[df['in_degree'] > in_degree_threshold]
    return df, Adj


def _bokeh_colormap(series, cmap='viridis', stretch=True, mode=(lambda x: x)):
    if stretch:
        series = ((series - series.min()) /
                  (series.max() - series.min()))
    series = mode(series)
    colors_raw = cm.get_cmap(cmap)(series, bytes=True)
    colors_str = ['#%02x%02x%02x' % tuple(c[:3]) for c in colors_raw]
    return colors_str


def _argument_parser():
    parser = argparse.ArgumentParser('prin', description=__doc__)
    parser.add_argument('datafile', help='File containing network data.')
    parser.add_argument('-f', '--format', help='What format/parser to use.')
    parser.add_argument('-o', '--output-file', help='HTML file for plot.',
                        default='plot.html')
    parser.add_argument('-n', '--max-num-nodes', type=int, default=int(1e7),
                        help='Limit number of nodes read.')
    parser.add_argument('-t', '--in-degree-threshold', type=float, default=-1,
                        help='Do not plot nodes with smaller in-degree')
    parser.add_argument('-T', '--pagerank-threshold', type=float, default=-1,
                        help='Do not plot nodes with smaller pagerank. '
                             'This threshold is divided by the total number '
                             'of nodes.')
    parser.add_argument('-l', '--linear', action='store_false', dest='loglog',
                        default=True, help='Use a linear, not log-log, '
                                           'scale for the scatterplot')
    parser.add_argument('-d', '--damping', type=float, default=0.85,
                        help='Damping value for pagerank computation.')
    parser.add_argument('-O', '--spectral-offset', type=float, default=0.5,
                        help='Offset to apply to Laplacian diagonal for '
                             'eigenvalue stabilization.')
    parser.add_argument('-D', '--data-frame', type=str,
                        help='Save resulting data frame to this file.')
    return parser


def bokeh_plot(df, output='plot.html', color=None, loglog=True):
    tooltip = [('name', '@id'),
               ('description', '@description'),
               ('pagerank', '@pagerank'),
               ('in-degree', '@in_degree')]
    if color is not None:
        df['color'] = _bokeh_colormap(df[color])
    source = ColumnDataSource(df)
    bplot.output_file(output)
    hover = HoverTool(tooltips=tooltip)
    tools = [t() for t in TOOLS] + [hover]
    if loglog:
        yaxis = xaxis = 'log'
    else:
        yaxis = xaxis = 'linear'
    pagerank = bplot.figure(tools=tools,
                            x_axis_type=xaxis, y_axis_type=yaxis)
    if color is not None:
        pagerank.circle('in_degree', 'pagerank', color='color', source=source)
    else:
        pagerank.circle('in_degree', 'pagerank', source=source)
    bplot.show(pagerank)


def serve(datasource, output='plot.html'):
    columns = list(datasource.keys())
    positive_vars = [k for k in columns if np.all(datasource[k] > 0)]
    x = Select(title='X-Axis', value='in_degree',
               options=columns)
    y = Select(title='Y-Axis', value='pagerank',
               options=columns)
    size = Select(title='Size', value='None',
                  options=['None'])


from matplotlib import pyplot as plt
from matplotlib import colors


def plot_connectome(neuron_x, neuron_y, links, labels, types):
    colormap = colors.ListedColormap([[0.   , 0.447, 0.698],
                                      [0.   , 0.62 , 0.451],
                                      [0.835, 0.369, 0.   ]])
    # plot neuron locations:
    plt.scatter(neuron_x, neuron_y, c=types, cmap=colormap,
                edgecolors='face', zorder=1)

    # add text labels:
    for x, y, label in zip(neuron_x, neuron_y, labels):
        plt.text(x, y, '  ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=5, zorder=2)

    # plot links
    pre, post = np.nonzero(links)
    for src, dst in zip(pre, post):
        plt.plot(neuron_x[[src, dst]], neuron_y[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)

    plt.show()


def plot_dependencies(xs, ys, A, names, values=None, subsample=10,
                      attenuation=2):
    if values is None:
        values = np.full(len(xs), 1/len(xs))
    values = values ** (1 / attenuation)
    values /= np.sum(values)  # normalize to sum to 1 for probabilities
    if subsample:
        indices = np.random.choice(np.arange(len(xs)), p=values,
                                   size=len(xs) // subsample, replace=False)
    else:
        indices = np.arange(len(xs))
    values = values[indices]
    indices = indices[np.argsort(values)]  # plot low values first
    values = np.sort(values) / np.max(values)  # normalize to max-1 for scaling
    xs = xs[indices]
    ys = ys[indices]
    A = A[indices][:, indices]
    names = [names[i] for i in indices]
    colormap = plt.cm.plasma_r
    plt.scatter(xs, ys, s=values*50, c=values,
                cmap=colormap, alpha=0.5, zorder=1)

    # add text labels
    for x, y, label, val in zip(xs, ys, names, values):
        plt.text(x, y, '   ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12 * val, alpha=val, zorder=2)

    pre, post = np.nonzero(A)
    for src, dst in zip(pre, post):
        plt.plot(xs[[src, dst]], ys[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)
    plt.show()


def main(argv):
    args = _argument_parser().parse_args(argv)
    if args.data_frame is not None and os.path.exists(args.data_frame):
        df = feather.read_dataframe(args.data_frame)
    else:
        from . import parsers
        parser = getattr(parsers, args.format).parser
        print('reading network data')
        network = parser(args.datafile, max_num_nodes=args.max_num_nodes)
        print('extracting data')
        df = network_properties(network,
                                in_degree_threshold=args.in_degree_threshold,
                                pagerank_threshold=args.pagerank_threshold,
                                damping=args.damping)
    if args.data_frame is not None:
        feather.write_dataframe(df, args.data_frame)
    print('preparing plots')
    bokeh_plot(df, output=args.output_file, loglog=args.loglog)


if __name__ == '__main__':
    main(sys.argv[1:])
