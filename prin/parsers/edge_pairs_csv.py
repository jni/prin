import networkx as nx
import toolz as tz
from toolz import curried as c


c_open = tz.curry(open)
c_split = tz.curry(str.split)


def parser(filename, *args, **kwargs):
    g = nx.DiGraph()
    tz.pipe(filename, c_open(mode='r'),
            c.map(str.strip),
            c.map(c_split(sep=',')),
            g.add_edges_from)
    return g
