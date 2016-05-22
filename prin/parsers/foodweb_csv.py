import numpy as np
import networkx as nx

def parser(filename_root, *args, **kwargs):
    filename_network = filename_root + '.csv'
    network_np = np.loadtxt(filename_network, 'int', delimiter=',')
    network_nx = nx.from_numpy_matrix(network_np, create_using=nx.DiGraph())
    filename_species = filename_root + '.species.txt'
    try:
        with open(filename_species, 'r') as fin:
            mapping = dict(enumerate(fin))
        nx.relabel_nodes(network_nx, mapping=mapping, copy=False)
    except FileNotFoundError:
        print('No species name file found: %s' % filename_species)
    return network_nx