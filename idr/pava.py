import numpy as np
from sklearn.isotonic import IsotonicRegression

def pavaDec(cpY, thresholds, weights):
    """
    Compute one-dimensional isotonic distributional regression (IDR) using
    the Pool-Adjacent Violators Algorithm (PAVA): For two vectors x and y,
    solve the problem
               minimize sum( (1{y_i <= y_k} - p_i)^2 )
               subject to p_i >= p_j if x_i <= x_j
      for each k, where 1{A} = 1 if A is TRUE and 0 otherwise. If the values of
      x are not distinct, then there are
           xx_1 < xx_2 < ... < xx_m
      such that all entries x_i are contained in {xx_1, ..., xx_m}. Denote by
      cpY[[i]] the y values belongig to xx[i]. Then the above minimizaion
      problem is equivalent to
               minimize sum(w * (mean(cpY[[j]] <= y_k) - q_j)^2 )
               subject to q_1 >= q_2 >= ... >= q_m,
      where w = (length(crpY[[1]]), ..., length(cpY[[m]])).
      This can be solved by applying antitonic PAVA to the vectors
               (mean(cpY[[1]] <= y_k), ...., mean(cpY[[m]] <= y_k))
      with weigths w.
     
    Parameters
    ----------
    cpY : list
    thresholds : one dimensional np.array
    weights : one dimensional np.array

    """
    n = len(cpY)
    m = len(thresholds)-1
    yin = np.zeros((n,m))
    for i in range(n):
        tmp = cpY[i]
        for j in range(m):
            yin[i,j] = np.mean(tmp <= thresholds[j])
    ir = IsotonicRegression(increasing = False)
    x = np.arange(n)
    out = np.zeros((n,m))
    for k in range(m):
        y = yin[:,k]
        out[:,k] = ir.fit_transform(x, y, sample_weight = weights)
        #out.append(ir.fit_transform(x, y, sample_weight = weights))
    return(out)


def pavaCorrect(cdf):
    
    """
      Apply the Pool-Adjacent Violators Algorithm (PAVA) to the rows of a
      numeric matrix y. This code is adapted from the R code by Lutz Duembgen
      (2008) at
      http://www.imsv.unibe.ch/about_us/files/lutz_duembgen/software/index_eng.html
     
     
    Parameters
    ----------
    cpY : list
    """
    n = cdf.shape[0]
    m = cdf.shape[1]
    x = np.arange(n)
    out = np.zeros((n,m))
    for k in range(m):
        y = cdf[:,k]
        ir = IsotonicRegression(increasing = False)
        out[:,k] = ir.fit_transform(x, y)
    return(out)
