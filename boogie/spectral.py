from scipy import sparse
import numpy as np
import networkx as nx
import warnings


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


def node_coordinates(graph, remove_nodes=None, nodelist=None):
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
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=names)
    C = (A + A.T) / 2
    degrees = np.ravel(C.sum(axis=0))
    D = sparse.diags([degrees], [0]).tocsr()
    L = D - C
    x, y = affinity_view(D, L)
    z = processing_depth(A, C, L)
    return x, y, z, A, names


