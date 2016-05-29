"""
Compute pagerank on a bunch of datasets and plot it against the in-degree.

The point is to find points that have high PR but relatively low in-degree
and hold them up as examples of PR success.

Based on experience these are hard to come by. =P
"""
import sys
import argparse

import numpy as np
import feather

import toolz as tz
from toolz import curried as c

from matplotlib import cm
import networkx as nx

from scipy import sparse
from skimage import io, filters, measure, morphology
import pandas as pd
from sklearn import decomposition, manifold

from bokeh.models import (LassoSelectTool, PanTool,
                          ResizeTool, ResetTool,
                          HoverTool, WheelZoomTool)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResizeTool, ResetTool]
from bokeh.models import ColumnDataSource
from bokeh import plotting as bplot
#from bokeh.plotting import figure, gridplot, output_file, show


def pagerank_power(Trans, damping=0.85, max_iter=int(1e5)):
    n = Trans.shape[0]
    dangling = np.ravel(Trans.sum(axis=0) == 0) * (1/n)
    r0 = np.full(n, 1/n)
    r = r0
    beta = (1 - damping) / n
    for _ in range(max_iter):
        rnext = damping * (Trans @ r + dangling @ r) + beta
        if np.allclose(rnext, r):  # converged!
            return rnext
        else:
            r = rnext
    return r


def compute_pagerank(network : nx.DiGraph, damping : float=0.85):
    Adj = nx.to_scipy_sparse_matrix(network, dtype='float', format='csr')
    deg = np.ravel(Adj.sum(axis=1))
    Dinv = sparse.diags(1 / deg)
    Trans = (Dinv @ Adj).T
    pr = pagerank_power(Trans, damping=damping)
    return pr


def network_properties(network : nx.DiGraph,
                       in_degree_threshold : float = -1,
                       pagerank_threshold : float = -1,
                       damping : float = 0.85,
                       values=[]) -> pd.DataFrame:
    conn = max(nx.connected_components(network.to_undirected()), key=len)
    conn = nx.subgraph(network, conn)
    pr = compute_pagerank(conn, damping=damping)
    names = nx.nodes(conn)
    indeg = [conn.in_degree(n) for n in names]
    odeg = [conn.out_degree(n) for n in names]
    description = [conn.node[n].get('description', n) for n in names]
    data = {'id': names,
            'in_degree': indeg,
            'out_degree': odeg,
            'pagerank': pr,
            'description': description}
    df = pd.DataFrame(data, index=names)
    df = df[df['pagerank'] > pagerank_threshold / len(names)]
    df = df[df['in_degree'] > in_degree_threshold]
    return df


def _bokeh_colormap(series, cmap='viridis', stretch=True):
    if stretch:
        series = ((series - series.min()) /
                  (series.max() - series.min()))
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


def main(argv):
    args = _argument_parser().parse_args(argv)
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
