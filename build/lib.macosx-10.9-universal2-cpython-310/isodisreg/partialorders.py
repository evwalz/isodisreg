import numpy as np
import scipy.stats
import pandas as pd
pd.options.mode.chained_assignment = None

def comp_ord(X):
    """Componentwise partial order on rows of array x.

    Compares the columns x[j, j] of a matrix x in the componentwise
    order.


    Parameters
    ----------
    x : np.array
        Two-dimensional array with at least two columns.
    Returns
    -------
    paths : np.array
        Two-column array, containing in the first coolumns the indices
        of the rows of x that are smaller in the componentwise order,
        and in the second column the indices of the corresponding
        greater rows.
    col_order : np.array
        Array of the same dimension of x, containing in each column the
        order of the column in x.
    """
    X = np.asarray(X)
    if X.ndim != 2 | X.shape[0] < 2:
        raise ValueError("X should have at least two rows")
    Xt = X.transpose()
    m = Xt.shape[1]
    d = Xt.shape[0]
    colOrder = np.argsort(Xt, axis=1)
    ranks = np.apply_along_axis(scipy.stats.rankdata, 1, Xt, method='max')
    smaller = []
    greater = []
    for k in range(m):
        nonzeros = np.full((m), False)
        nonzeros[colOrder[0,0:ranks[0,k]]] = True
    
        for l in range(1,d):
            if ranks[l,k]<m:
                nonzeros[colOrder[l,ranks[l,k]:m]] = False
        nonzeros = np.where(nonzeros)[0]
        n_nonzeros = nonzeros.shape[0]
        smaller.extend(nonzeros)
        greater.extend([k]*n_nonzeros)
    paths = np.vstack([smaller, greater]) 
    return paths, colOrder.transpose()


def tr_reduc(paths, n):
    """Transitive reduction of path matrix.

    Transforms transitive reduction of a directed acyclic graph.

    Parameters
    ----------
    x : np.array
        Two-dimensional array containing the indices of the smaller
        vertices in the first row and the indices of the
        greater vertices in the second row.
    Returns
    -------

    """
    edges = np.full((n, n), False)
    edges[paths[0], paths[1]] = True
    np.fill_diagonal(edges, False)
    for k in range(n):
        edges[np.ix_(edges[:, k], edges[k])] = False
    edges = np.array(np.nonzero(edges))
    return edges


def neighbor_points(x, X, order_X):
    """    
    Neighbor points with respect to componentwise order
    
    Parameters
    ----------
    x : np.array
        Two-dimensional array 
    X : Two dimensional array with at least to columns
    order_X : output of function compOrd(X)

    Returns
    -------
    list given for each x[i,] the indices 
    of smaller and greater neighbor points within the rows of X

    """
    X = np.asarray(X)
    x = np.asarray(x)                
    col_order = order_X[1]

    nx = x.shape[0]
    k = x.shape[1]
    n = X.shape[0]
    ranks_left = np.zeros((nx,k))
    ranks_right = np.zeros((nx,k))
    for j in range(k):
        ranks_left[:,j] = np.searchsorted(a = X[:,j], v = x[:,j], sorter = col_order[:,j])
        ranks_right[:,j] = np.searchsorted(a = X[:,j], v = x[:,j], side = "right", sorter = col_order[:,j])

    x_geq_X = np.full((n, nx), False)
    x_leq_X = np.full((n, nx), True)
    for i in range(nx):
        if ranks_right[i,0] > 0:
            x_geq_X[col_order[0:int(ranks_right[i,0]),0],i] = True
        if ranks_left[i,0] > 0:
            x_leq_X[col_order[0:int(ranks_left[i,0]),0],i] = False
        for j in range(1, k):
            if ranks_right[i,j] < n:
                x_geq_X[col_order[int(ranks_right[i,j]):n,j],i] = False
            if ranks_left[i,j] > 0:
                x_leq_X[col_order[0:int(ranks_left[i,j]),j],i] = False
    paths = np.full((n,n), False)
    paths[order_X[0][0], order_X[0][1]] = True
    np.fill_diagonal(paths, False)  

    for i in range(n):
        x_leq_X[np.ix_(paths[i,:], x_leq_X[i, :])] = False
        x_geq_X[np.ix_(paths[:, i], x_geq_X[i, :])] = False

    smaller = []
    greater = []

    for i in range(nx):
        smaller.append(x_geq_X[:,i].nonzero()[0])
        greater.append(x_leq_X[:,i].nonzero()[0])
        
    return smaller, greater
