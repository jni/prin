"""
Compute pagerank on a bunch of datasets and plot it against the in-degree.

The point is to find points that have high PR but relatively low in-degree
and hold them up as examples of PR success.

Based on experience these are hard to come by. =P
"""
import os
import sys
import tempfile
import pathlib

import numpy as np

from matplotlib import cm

from scipy import ndimage as ndi
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


def bokeh_plot(df):
    tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="60" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@source_filenames</span>
            </div>
        </div>
              """
    colors_raw = cm.viridis((df['time'] - df['time'].min()) /
                            (df['time'].max() - df['time'].min()), bytes=True)
    colors_str = ['#%02x%02x%02x' % tuple(c[:3]) for c in colors_raw]
    df['color'] = colors_str
    source = ColumnDataSource(df)
    bplot.output_file('plot.html')
    hover = HoverTool(tooltips=tooltip)
    tools = [t() for t in TOOLS] + [hover]
    pagerank = bplot.figure(tools=tools)
    pagerank.circle('in-degree', 'pagerank', color='color', source=source)
    tsne = bplot.figure(tools=tools1)
    tsne.circle('tSNE-0', 'tSNE-1', color='color', source=source)
    p = bplot.gridplot([[pca, tsne]])
    bplot.show(p)


def main(argv):
    print('reading images')
    images = io.imread_collection(argv[1:],
                                  conserve_memory=False, plugin='tifffile')
    images = normalize_images(images)
    print('extracting data')
    table, df, weights = extract_properties_multi_image(images)

    print('preparing plots')
    bokeh_plot(df)


if __name__ == '__main__':
    main(sys.argv)
