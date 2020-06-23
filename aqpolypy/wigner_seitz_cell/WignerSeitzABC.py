"""
:module: WignerSeitzABC
:platform: Unix, Windows, OS
:synopsis: Defines an abstract unit class

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
..
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate


class WiSe(ABC):

    """ Defines a general Wigner Seitz cell"""

    def __init__(self, dim, n_distinct_wedges):
        """
        constructor

        :param dim: plane is d=1, cylinder d=2, sphere d=3
        :param n_distinct_wedges: number of distinct wedges
        """

        self.dim = dim
        self.distinct_wedges = n_distinct_wedges
        self.wedges = n_distinct_wedges*[None]
        self.num_wedges = n_distinct_wedges*[None]

    @abstractmethod
    def name(self):
        """
        name of the wigner seitz cell

        :return: name of the wigner-seitz cell
        """

    def s_angle_tot(self):
        """
        Computes in d=2(3), the angle(solid angle) spanned ed by all wedges

        :return: angle, solid angle (float)
        """

        return np.sum(self.s_angle()*np.array(self.num_wedges))

    def s_angle(self):
        """
        Computes in d=2(3), the angle(solid angle) spanned ed by each wedge

        :return: angle, solid angle (ndarray)
        """

        s_ang = np.zeros(self.distinct_wedges)
        if self.dim == 2:
            def fun(x):
                return 1
            for ind, wg in enumerate(self.wedges):
                s_ang[ind] = integrate.quad(fun, wg[0], wg[1])
        elif self.dim == 3:
            def fun(y, x):
                return np.sin(y)
            for ind, wg in enumerate(self.wedges):
                a, b = wg[0]
                f1, f2 = wg[1]
                s_ang[ind] = integrate.dblquad(fun, a, b, f1, f2)[0]
        else:
            s_ang = np.array([1])

        return s_ang

    def vol_tot(self, dn, rad=0):
        """
        computes in d=2,3 the area (volume) of the Wigner-Seitz cell

        :param dn: radius of the inscribed sphere (smallest of all distances)
        :param rad: radius of a cylinder(sphere) contained and whose center is at the origin of the wigner-seitz cell
        :return: area, volume (float)
        """

        if rad > dn:
            return 0

        return np.sum(self.vol(dn, rad) * np.array(self.num_wedges))

    def vol(self, dn, rad=0):
        """
        computes in d=2,3 the area (volume) of each wedge within the Wigner-Seitz cell

        :param dn: radius of the inscribed sphere (smallest of all distances)
        :param rad: radius of a cylinder(sphere) contained and whose center is at the origin of the wigner-seitz cell
        :return: area, volume (ndarray)
        """

        if rad > dn:
            return 0

        ws_vol = np.zeros(self.distinct_wedges)

        def h(x):
            return (dn+(1-np.cos(x))*rad)/np.cos(x)

        def ze(x):
            return 0

        if self.dim == 2:
            def fun(x, y):
                return y+rad
            for ind, wg in enumerate(self.wedges):
                a, b = wg[0]
                ws_vol[ind] = integrate.dblquad(fun, a, b, ze, h)
        elif self.dim == 3:
            def fun(z, y, x):
                return np.sin(y)*(z+rad)**2

            def h3(x, y):
                return h(y)

            def z3(x, y):
                return ze(y)

            for ind, wg in enumerate(self.wedges):
                a, b = wg[0]
                f1, f2 = wg[1]
                val = integrate.tplquad(fun, a, b, f1, f2, z3, h3)
                ws_vol[ind] = val[0]
        else:
            ws_vol[:] = dn-rad

        return ws_vol
