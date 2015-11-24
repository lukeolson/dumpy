import numpy as np
import scipy.sparse as sparse

def shortlyweller(msh):
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

    # [  N  ]
    # [W C E]
    # [  S  ]

    # list of d.o.f.
    J = np.arange(n, dtype=int)

    JnotN = J[np.logical_not(msh.IN)]
    JnotS = J[np.logical_not(msh.IS)]
    JnotE = J[np.logical_not(msh.IE)]
    JnotW = J[np.logical_not(msh.IW)]

    # for each d.o.f. k, add weights if they do not connect to a boundary
    AA = np.hstack((centerweight,
                    northweight[JnotN],
                    southweight[JnotS],
                    eastweight[JnotE],
                    westweight[JnotW]))

    # add the row
    AI = np.hstack((J,
                    JnotN,
                    JnotS,
                    JnotE,
                    JnotW))

    # find the column
    # for JnotN for example, find the global index of JnotN to the north
    #          I2D[0][JnotN], I2D[1][JnotN]   gives the 2D indices
    #          I2D[0][JnotN], I2D[1][JnotN]-1 gives the 2D indices to the north
    # indexmap[(I2D[0][JnotN], I2D[1][JnotN]-1)] gives the global indices to the north
    AJ = np.hstack((indexmap[(I2D[0][J],I2D[1][J])],
                    indexmap[(I2D[0][JnotN]+1, I2D[1][JnotN])],
                    indexmap[(I2D[0][JnotS]-1, I2D[1][JnotS])],
                    indexmap[(I2D[0][JnotE], I2D[1][JnotE]+1)],
                    indexmap[(I2D[0][JnotW], I2D[1][JnotW]-1)]))

    print(msh.IN)
    print(msh.IS)
    print(msh.IE)
    print(msh.IW)
    print(indexmap[(I2D[0][J],I2D[1][J])])
    print(indexmap[(I2D[0][JnotN]+1, I2D[1][JnotN])])
    print(indexmap[(I2D[0][JnotS]-1, I2D[1][JnotS])])
    print(indexmap[(I2D[0][JnotE], I2D[1][JnotE]+1)])
    print(indexmap[(I2D[0][JnotW], I2D[1][JnotW]-1)])

    A = sparse.coo_matrix((AA, (AI, AJ))).tocsr()
    return A
