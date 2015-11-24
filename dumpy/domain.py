import numpy as np
from scipy.optimize import bisect

class mesh:
    """
    attributes
    ---

    X, Y : array
      n x 1 list of x, y cooridnates of a regular grid that covers this domain

    I : array
      array of points interior to the domain (for which a discretization is sought

    xybdy : array
      list of points on the boundary.  xybdy is not in X, Y

    IN, IS, IE, IW : array
      subset of [0,len(I)] on the boundaries

    dN, dS, dE, dW : array
      distance to the boundary from each IN, IS, IE, IW points
    """

    nx = 1
    ny = 1
    hx = 0.0
    hy = 0.0

    X = None
    Y = None

    boundary_set = False

    def __init__(self):
        pass

    def boundary(self):
        pass

    def set_boundary(self, name):

        if name == 'circle':
            self.boundary = lambda x, y: x**2 + y**2 - 1.0
            self.boundary_set = True
        elif callable(name):
            self.boundary = name
            self.boundary_set = True

    def set_mesh(self, n):

        if not self.boundary_set:
            raise Error('need to set_boundary description first')

        nx = n
        ny = n
        hx = 2.0 / (nx - 1)
        hy = 2.0 / (ny - 1)

        X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
        X = X.ravel()
        Y = Y.ravel()

        I = np.where(self.boundary(X, Y) < 0)[0]
        IN = np.zeros((I.shape[0],), dtype=bool)
        IS = np.zeros((I.shape[0],), dtype=bool)
        IE = np.zeros((I.shape[0],), dtype=bool)
        IW = np.zeros((I.shape[0],), dtype=bool)

        dN = hy * np.ones((I.shape[0],), dtype=bool)
        dS = hy * np.ones((I.shape[0],), dtype=bool)
        dE = hx * np.ones((I.shape[0],), dtype=bool)
        dW = hx * np.ones((I.shape[0],), dtype=bool)

        for i in range(len(I)):
            x, y = X[I[i]], Y[I[i]]

            boundaryx = lambda xx: self.boundary(xx, y)
            boundaryy = lambda yy: self.boundary(x, yy)

            if self.boundary(x, y + hy) > 0:
                IN[i] = True
                dN[i] = abs(y - bisect(boundaryy, y, y + hy))

            if self.boundary(x, y - hy) > 0:
                IS[i] = True
                dS[i] = abs(y - bisect(boundaryy, y, y - hy))

            if self.boundary(x + hx, y) > 0:
                IE[i] = True
                dE[i] = abs(x - bisect(boundaryx, x, x + hx))

            if self.boundary(x - hx, y) > 0:
                IW[i] = True
                dW[i] = abs(x - bisect(boundaryx, x, x - hx))

        try:
            assert(len(np.where(IN)) == len(np.where(dN < hy)))
        except AssertionError:
            print('Problem finding distances to the boundary')
            raise

        att = {'X': X, 'Y': Y,
               'nx': nx, 'ny': ny, 'hx': hx, 'hy': hy,
               'I': I,
               'IN': IN, 'IS': IS, 'IE': IE, 'IW': IW,
               'dN': dN, 'dS': dS, 'dE': dE, 'dW': dW,
              }

        for k in att:
            setattr(self, k, att[k])

if __name__ == '__main__':

    run1 = mesh()

    run1.set_boundary(name='circle')
    run1.set_mesh(16)

    import matplotlib.pyplot as plt

    plt.plot(run1.X, run1.Y, 'o', clip_on=False);
    plt.plot(run1.X[run1.I], run1.Y[run1.I],
             'r*', clip_on=False, ms=10, label='interior')
    plt.plot(run1.X[run1.I[run1.IN]], run1.Y[run1.I[run1.IN]],
             'mo', clip_on=False, ms=15, label='north',
             mfc='None', mew=2, mec='m')
    plt.plot(run1.X[run1.I[run1.IS]], run1.Y[run1.I[run1.IS]],
             'yo', clip_on=False, ms=15, label='south',
             mfc='None', mew=2, mec='y')
    plt.plot(run1.X[run1.I[run1.IE]], run1.Y[run1.I[run1.IE]],
             'gs', clip_on=False, ms=10, label='east',
             mfc='None', mew=2, mec='g')
    plt.plot(run1.X[run1.I[run1.IW]], run1.Y[run1.I[run1.IW]],
             'cs', clip_on=False, ms=10, label='west',
             mfc='None', mew=2, mec='c')
    plt.legend()

    plt.show()
