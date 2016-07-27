from scipy import sparse
import numpy as np
import networkx as nx
import warnings
import itertools


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



def affinity_view(D, L):
    Dinv2 = D.copy()
    Dinv2.data = Dinv2.data ** (-.5)
    Q = Dinv2 @ L @ Dinv2
    eigvals, vec = sparse.linalg.eigsh(Q, k=3, which='SM')
    _, x, y = (Dinv2 @ vec).T
    return x, y


def processing_depth(A, C, L):
    b = C.multiply((A - A.T).sign()).sum(axis=1)
    z, error = sparse.linalg.isolve.cg(L, b, maxiter=int(1e4))
    if error > 0:
        warnings.warn('CG convergence failed after %s iterations' % error)
    elif error < 0:
        warnings.warn('CG illegal input or breakdown')
    return z


def node_coordinates(graph, remove_nodes=None, nodelist=None, offset=0):
    conn = max(nx.connected_components(graph.to_undirected()),
               key=len)
    subgraph = graph.subgraph(conn)
    if remove_nodes is not None:
        subgraph.remove_nodes_from(remove_nodes)
    subgraph.remove_edges_from(subgraph.selfloop_edges())
    if nodelist is None:
        names = subgraph.nodes()
    else:
        names = nodelist
    Adj = nx.to_scipy_sparse_matrix(subgraph, nodelist=names)
    Conn = (Adj + Adj.T) / 2 + sparse.diags(np.full(Adj.shape[0], offset))
    degrees = np.ravel(Conn.sum(axis=0))
    Deg = sparse.diags([degrees], [0]).tocsr()
    Lap = Deg - Conn
    x, y = affinity_view(Deg, Lap)
    z = processing_depth(Adj, Conn, Lap)
    return x, y, z, Adj, names


def coo_mat_concat(matrices, spformat='csr'):
    ii = []
    jj = []
    dat = []
    locr = 0
    locc = 0
    for mat in matrices:
        coo = sparse.coo_matrix(mat, shape=mat.shape)
        ii.append(coo.row)
        jj.append(coo.col)
        dat.append(coo.data)
        locr += coo.shape[0]
        locc += coo.shape[1]
    coo = sparse.coo_matrix((np.concatenate(dat),
                             (np.concatenate(ii), np.concatenate(jj))))
    out = getattr(coo, 'to' + spformat)()
    return out


def node_coordinates_robust(graph):
    xs, ys, zs, As, namess = [], [], [], [], []
    for cc in nx.connected_components(graph.to_undirected()):
        if len(cc) == 1:
            x, y, z = [0], [0], [0]
            A = np.array([[0]])
            names = list(cc)
        elif len(cc) == 2:
            x, y, z = [0, 1], [0, 1], [0, 1]
            n1, n2 = list(cc)
            A = np.array([[0, 1], [1, 0]]) * graph[n1][n2].get('weight', 1)
            names = list(cc)
        else:
            x, y, z, A, names = node_coordinates(nx.subgraph(graph, cc))
        xs.append(x)
        ys.append(y)
        zs.append(z)
        As.append(A)
        namess.append(names)
    for coord in [xs, ys, zs]:
        loc = 0
        for i, arr in enumerate(coord):
            arr = np.asanyarray(arr)
            scale = np.sqrt(arr.size - 0.99)
            coord[i] = ((arr - np.min(arr)) / np.max(arr) *
                        scale + loc)
            loc += 1.05 * scale
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    A = coo_mat_concat(As)
    names = list(itertools.chain(*namess))
    return x, y, z, A, names
