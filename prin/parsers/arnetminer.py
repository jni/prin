import io
import re
import networkx as nx
import numpy as np
import toolz as tz
from toolz import curried as c


flagdict = {
    '*': 'title',
    '@': 'authors',
    't': 'year',
    'c': 'journal',
    'index': 'index',
    '%': 'reference',
    '!': 'abstract'
}

# Regular expression to parse a field in the ArnetMiner flat file
# - Find a # at the start of the file but ignore it
# - Capture the next character or character group as the line 'flag',
#   indicating the content of the line (see `flagdict`, above)
# - Ignore intervening whitespace
# - Capture the rest of the line into the 'data' group.
rexp = r'^#(?P<flag>\*|@|t|c|index|%|!)\s*(?P<data>.*)$'


MAXINT = np.iinfo(np.int32).max


def _describe_record(rec):
    authors = [auth.split()[-1] for auth in rec['authors']]
    if len(authors) == 0:
        authors = ''
    elif len(authors) == 1:
        authors = authors[0]
    elif len(authors) == 2:
        authors = ' & '.join(authors)
    else:
        authors = authors[0] + ' et al.'
    desc = authors + ', ' + rec['year'] + '. ' + rec['title']
    return desc


def _default_record():
    record = {}
    for key in ['title', 'year', 'journal', 'abstract']:
        record[key] = ''
    for key in ['authors', 'references']:
        record[key] = []
    return record


def get_record(linestup, ignore_abstracts=True):
    record = _default_record()
    for line in linestup:
        m = re.match(rexp, line)
        if m is None:
            raise ValueError('failed regex matching for line:\n%s' % line)
        flag, data = m.groups()
        kind = flagdict[flag]
        if kind == 'abstract' and ignore_abstracts:
            continue
        if kind == 'reference':
            record['references'].append(data)
        elif kind == 'authors':
            record[kind] = data.split(',')
        else:
            record[kind] = data
    record['description'] = _describe_record(record)
    return record


def _line_is_empty(line):
    return len(line.strip()) == 0


def _decode(bytesline):
    return bytesline.decode('utf-8')


def txt_parser(filelike, max_num_nodes=MAXINT):
    if isinstance(filelike, io.IOBase):
        fileobj = filelike
    else:  # assume filename
        fileobj = open(filelike, 'rb')
    g = nx.DiGraph()
    # this pipe assumes there are no empty lines at the start of the file
    records = tz.pipe(fileobj,
                      c.map(_decode),
                      c.partitionby(_line_is_empty),  # split on empty lines
                      c.take_nth(2),  # discard those empty lines
                      c.take(max_num_nodes),
                      c.map(get_record))
    for record in records:
        g.add_node(record['index'], attr_dict=record)
        for reference in record.get('references', []):
            g.add_edge(record['index'], reference)
    return g


def tar_parser(filename, max_num_nodes=MAXINT):
    import tarfile
    tf = tarfile.open(filename, 'r')
    inner = tf.getnames()[0]  # assume single file
    fileobj = tf.extractfile(inner)
    g = txt_parser(fileobj, max_num_nodes)
    return g


def parser(filename, max_num_nodes=MAXINT):
    if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        g = tar_parser(filename, max_num_nodes)
    elif filename.endswith('.txt'):
        g = txt_parser(filename, max_num_nodes)
    else:
        raise ValueError('Unknown file extension for ArnetMiner in %s, '
                         'use ".tar.gz", ".tgz", or ".txt".' % filename)
    return g