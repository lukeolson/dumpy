import numpy as np
import scipy.sparse as sparse

def shortlyweller(msh, ):
    """
    shortlyweller discretization for nonsmooth boundary

    input
    ---

    msh: object
        comain.mesh() structure

    output
    ---
    A: sparse

    """

    n = len(msh.I)
    #nnzmax = 5 * n

    #AA = np.zeros((nnzmax,))
    #AI = -np.ones((nnzmax,), dtype=int)
    #AJ = -np.ones((nnzmax,), dtype=int)

    hN = np.abs(msh.dN)
    hS = np.abs(msh.dS)
    hE = np.abs(msh.dE)
    hW = np.abs(msh.dW)

    northweight = -2.0 / (hN * (hN + hS))
    southweight = -2.0 / (hS * (hN + hS))
    eastweight = -2.0 / (hE * (hW + hE))
    westweight = -2.0 / (hW * (hW + hE))
    centerweight = 2.0 / (hW * hE) + 2.0 / (hN * hS)

    indexmap = msh.indexmap
    I2D = msh.I2D

    AA = np.hstack((centerweight,
                    northweight[np.logical_not(msh.IN)],
                    southweight[np.logical_not(msh.IS)],
                    eastweight[np.logical_not(msh.IE)],
                    westweight[np.logical_not(msh.IW)]))

    J = np.arange(n, dtype=int)

    AI = np.hstack((J,
                    J[np.logical_not(msh.IN)],
                    J[np.logical_not(msh.IS)],
                    J[np.logical_not(msh.IE)],
                    J[np.logical_not(msh.IW)]))
    AJ = np.hstack((indexmap[I2D[0][J],I2D[1][J]],
                    indexmap[I2D[0][J[np.logical_not(msh.IN)]],
                             I2D[1][J[np.logical_not(msh.IN)]]-1],
                    indexmap[I2D[0][J[np.logical_not(msh.IS)]],
                             I2D[1][J[np.logical_not(msh.IS)]]+1],
                    indexmap[I2D[0][J[np.logical_not(msh.IE)]],
                             I2D[1][J[np.logical_not(msh.IE)]]+1],
                    indexmap[I2D[0][J[np.logical_not(msh.IW)]],
                             I2D[1][J[np.logical_not(msh.IW)]]-1]))

    print(AJ)
    A = sparse.coo_matrix((AA, (AI, AJ))).tocsr()

    #uh = sla.spsolve(A, b)
