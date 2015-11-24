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
      (signed) distance to the boundary from each IN, IS, IE, IW points

    functions
    ---

    set_boundary : function
      a function of x, y that describes f(x,y) = 0, the boundary curve
      f(x, y) < 0 is inside the curve
      f(x, y) > 0 is outside the curve
    """

    nx = 1
    ny = 1
    hx = 0.0
    hy = 0.0

    X = None
    Y = None

    boundary_set = False

    def __init__(self, name, extent, nx, ny):
        self.set_boundary(name)
        self.set_mesh(extent, nx, ny)

    def _boundary(self):
        """
        a blank boundary function
        """
        pass

    def set_boundary(self, name):
        """
        a function of x, y that describes f(x,y) = 0, the boundary curve

        f(x, y) < 0 is inside the curve
        f(x, y) > 0 is outside the curve
        """

        if name == 'circle':
            self._boundary = lambda x, y: x**2 + y**2 - 1.0
            self.boundary_set = True
        if name == 'square':
            self._boundary = lambda x, y: np.maximum(np.abs(x), np.abs(y)) + 0.0*x - 1.0
            self.boundary_set = True
        elif callable(name):
            self._boundary = name
            self.boundary_set = True

    def set_mesh(self, extent, nx, ny):
        """
        sets a mesh that should overlap the boundary

        extent: array
            array of xmin, xmax, ymin, ymax for the mesh

        nx, ny: int
            mesh sizes
        """

        tol = 1e-14

        if not self.boundary_set:
            raise Error('need to set_boundary description first')

        xmin, xmax, ymin, ymax = extent
        hx = (xmax - xmin) / (nx - 1)
        hy = (ymax - ymin) / (ny - 1)

        X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                           np.linspace(ymin, ymax, ny))
        # keep 2D indexing
        I2D = np.where(self._boundary(X, Y) < -tol)
        I = np.ravel_multi_index(I2D, (nx,ny))

        n = len(I)
        indexmap = -np.ones(X.shape, dtype=int)
        indexmap[I2D] = np.arange(n, dtype=int)

        IN = np.zeros((n,), dtype=bool)
        IS = np.zeros((n,), dtype=bool)
        IE = np.zeros((n,), dtype=bool)
        IW = np.zeros((n,), dtype=bool)

        dN = hy * np.ones((n,))
        dS = hy * np.ones((n,))
        dE = hx * np.ones((n,))
        dW = hx * np.ones((n,))

        X = X.ravel()
        Y = Y.ravel()
        for i in range(len(I)):
            x, y = X[I[i]], Y[I[i]]

            boundaryx = lambda xx: self._boundary(xx, y)
            boundaryy = lambda yy: self._boundary(x, yy)

            if self._boundary(x, y + hy) > -tol:
                IN[i] = True
                dN[i] = bisect(boundaryy, y, y + 2*hy) - y

            if self._boundary(x, y - hy) > -tol:
                IS[i] = True
                dS[i] = bisect(boundaryy, y, y - 2*hy) - y

            if self._boundary(x + hx, y) > -tol:
                IE[i] = True
                dE[i] = bisect(boundaryx, x, x + 2*hx) - x

            if self._boundary(x - hx, y) > -tol:
                IW[i] = True
                dW[i] = bisect(boundaryx, x, x - 2*hx) - x

        try:
            assert(len(np.where(IN)) == len(np.where(dN < hy)))
        except AssertionError:
            print('Problem finding distances to the boundary')
            raise

        att = {'X': X, 'Y': Y,
               'nx': nx, 'ny': ny, 'hx': hx, 'hy': hy,
               'I': I, 'I2D': I2D,
               'IN': IN, 'IS': IS, 'IE': IE, 'IW': IW,
               'dN': dN, 'dS': dS, 'dE': dE, 'dW': dW,
               'indexmap': indexmap,
              }

        for k in att:
            setattr(self, k, att[k])

if __name__ == '__main__':

    nx=18
    ny=18
    run1 = mesh(name='circle', extent=[-2,2,-2,2], nx=nx, ny=ny)

    I = run1.I
    IN = run1.IN
    IS = run1.IS
    IE = run1.IE
    IW = run1.IW

    import disc
    A = disc.shortlyweller(run1)

    u = 1 - run1.X[I]**2 - run1.Y[I]**2
    f = 4*np.ones(run1.X[I].shape)

    import scipy.sparse.linalg as spla
    uh = spla.spsolve(A, f)

    import matplotlib.pyplot as plt

    plt.figure()
    uhgrid = np.zeros(run1.X.shape) * np.nan
    uhgrid[run1.I] = uh
    plt.pcolormesh(run1.X.reshape((nx,ny)), run1.Y.reshape((nx,ny)), uhgrid.reshape((nx,ny)))

    plt.figure()
    plt.plot(run1.X, run1.Y, 'o', clip_on=False);

    plt.plot(run1.X[I], run1.Y[I],
             'r*', clip_on=False, ms=10, label='interior')
    plt.plot(run1.X[I[IN]], run1.Y[I[IN]],
             'mo', clip_on=False, ms=15, label='north',
             mfc='None', mew=2, mec='m')
    plt.plot(run1.X[I[IS]], run1.Y[I[IS]],
             'yo', clip_on=False, ms=15, label='south',
             mfc='None', mew=2, mec='y')
    plt.plot(run1.X[I[IE]], run1.Y[I[IE]],
             'gs', clip_on=False, ms=10, label='east',
             mfc='None', mew=2, mec='g')
    plt.plot(run1.X[I[IW]], run1.Y[I[IW]],
             'cs', clip_on=False, ms=10, label='west',
             mfc='None', mew=2, mec='c')

    plt.contour(run1.X.reshape((nx,ny)),
                run1.Y.reshape((nx,ny)),
                run1._boundary(run1.X, run1.Y).reshape((nx,ny)),
                levels=[0])

    plt.plot(run1.X[I[IN]],
             run1.Y[I[IN]] + run1.dN[IN], 'k+', ms=10)
    plt.plot(run1.X[I[IS]],
             run1.Y[I[IS]] + run1.dS[IS], 'k+', ms=10)
    plt.plot(run1.X[I[IE]] + run1.dE[IE],
             run1.Y[I[IE]], 'k+', ms=10)
    plt.plot(run1.X[I[IW]] + run1.dW[IW],
             run1.Y[I[IW]], 'k+', ms=10)

    plt.legend()
    plt.show()
