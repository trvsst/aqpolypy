"""
:module: WaterWagner
:platform: Unix, Windows, OS
:synopsis: Derived water properties class utilizing IAPWS-95 formulation calculations

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added general calculations needed to derive the free energy
"""

import numpy as np
import aqpolypy.units.units as un
from scipy import optimize


class WaterWagner:
    """
    Water properties following the work of Wagner :cite:

    """

    def __init__(self, t, pa=1, d=float("NaN")):
        """
        constructor

        :param tk: temperature in kelvin
        :param pa: pressure in atmospheres
        :param d: density in SI
        :instantiate: temperature and pressure

        """
        self.t = t
        self.p = un.atm_2_pascal(pa)
        self.d = d

        self.tc = 647.096
        self.dc = 322
        self.R = 0.46151805

        self.delta = self.d / self.dc
        self.tau = self.tc / self.t

    def phi_naught(self):
        a = np.array([[-8.32044648201, 0.0],
                      [6.6832105268, 0.0],
                      [3.00632, 0.0],
                      [0.012436, 1.28728967],
                      [0.97315, 3.53734222],
                      [1.27950, 7.74073708],
                      [0.96956, 9.24437796],
                      [0.24873, 27.5075105]])

        def g(i, n):
            t2 = 0
            for x in range(i, n):
                t2 = t2 + (a[x][0] * np.log(1 - np.exp(-a[x][1] * self.tau)))
            return t2

        term_1 = (np.log(self.delta)) + a[0][0] + (a[1][0] * self.tau) + (a[2][0] * np.log(self.tau))
        term_2 = g(3, 8)

        phi_o = term_1 + term_2

        return phi_o

    def phi_r(self):
        a = np.array([[0.0, 1, -0.5, 0.12533547935523e-1],
                      [0.0, 1, 0.875, 0.78957634722828e1],
                      [0.0, 1, 1, -0.87803203303561e1],
                      [0.0, 2, 0.5, 0.31802509345418],
                      [0.0, 2, 0.75, -0.26145533859358],
                      [0.0, 3, 0.375, -0.78199751687981e-2],
                      [0.0, 4, 1, 0.88089493102134e-2],
                      [1, 1, 4, -0.66856572307965],
                      [1, 1, 6, 0.20433810950965],
                      [1, 1, 12, -0.66212605039687e-4],
                      [1, 2, 1, -0.19232721156002],
                      [1, 2, 5, -0.25709043003438],
                      [1, 3, 4, 0.16074868486251],
                      [1, 4, 2, -0.40092828925807e-1],
                      [1, 4, 13, 0.39343422603254e-6],
                      [1, 5, 9, -0.75941377088144e-5],
                      [1, 7, 3, 0.56250979351888e-3],
                      [1, 9, 4, -0.15608652257135e-4],
                      [1, 10, 11, 0.11537996422951e-8],
                      [1, 11, 4, 0.36582165144204e-6],
                      [1, 13, 13, -0.13251180074668e-11],
                      [1, 15, 1, -0.62639586912454e-9],
                      [2, 1, 7, -0.10793600908932],
                      [2, 2, 1, 0.17611491008752e-1],
                      [2, 2, 9, 0.22132295167546],
                      [2, 2, 10, -0.40247669763528],
                      [2, 3, 10, 0.58083399985759],
                      [2, 4, 3, 0.49969146990806e-2],
                      [2, 4, 7, -0.31358700712549e-1],
                      [2, 4, 10, -0.74315929710341],
                      [2, 5, 10, 0.47807329915480],
                      [2, 6, 6, 0.20527940895948e-1],
                      [2, 6, 10, -0.13636435110343],
                      [2, 7, 10, 0.14180634400617e-1],
                      [2, 9, 1, 0.83326504880713e-2],
                      [2, 9, 2, -0.29052336009585e-1],
                      [2, 9, 3, 0.38615085574206e-1],
                      [2, 9, 4, -0.20393486513704e-1],
                      [2, 9, 8, -0.16554050063734e-2],
                      [2, 10, 6, 0.19955571979541e-2],
                      [2, 10, 9, 0.15870308324157e-3],
                      [2, 12, 8, -0.16388568342530e-4],
                      [3, 3, 16, 0.43613615723811e-1],
                      [3, 4, 22, 0.34994005463765e-1],
                      [3, 4, 23, -0.76788197844621e-1],
                      [3, 5, 23, 0.22446277332006e-1],
                      [4, 14, 10, -0.62689710414685e-4],
                      [6, 3, 50, -0.55711118565645e-9],
                      [6, 6, 44, -0.19905718354408],
                      [6, 6, 46, 0.31777497330738],
                      [6, 6, 50, -0.11841182425981],
                      [0.0, 3, 0, -0.31306260323435e2, 20, 150, 1.21, 1],
                      [0.0, 3, 1, 0.31546140237781e2, 20, 150, 1.21, 1],
                      [0.0, 3, 4, -0.25213154341695e4, 20, 250, 1.25, 1]], dtype=object)

        def f(x):
            func = a[x][3] * self.delta ** a[x][1] * self.tau ** a[x][2]
            return func

        def h(x):
            func = np.exp(-a[x][4] * (self.delta - a[x][7]) ** 2 - a[x][5] * (self.tau - a[x][6]) ** 2)
            return func

        def g_1(i, n):
            t1 = 0
            for x in range(i, n):
                t1 = t1 + f(x)
            return t1

        def g_2(i, n):
            t2 = 0
            for x in range(i, n):
                t2 = t2 + f(x) * np.exp(-self.delta ** a[x][0])
            return t2

        def g_3(i, n):
            t3 = 0
            for x in range(i, n):
                t3 = t3 + (f(x) * h(x))
            return t3

        c = np.array([[3.5, 0.85, 0.2, -0.14874640856724, 28, 700, 0.32, 0.3],
                      [3.5, 0.95, 0.2, 0.31806110878444, 32, 800, 0.32, 0.3]])

        def theta(x):
            func = (1 - self.tau) + c[x][6] * ((self.delta - 1) ** 2) ** (1 / (2 * c[x][7]))
            return func

        def delta(x):
            func = theta(x) ** 2 + c[x][2] * ((self.delta - 1) ** 2) ** c[x][0]
            return func

        def psi(x):
            func = np.exp(-c[x][4] * (self.delta - 1) ** 2 - c[x][5] * (self.tau - 1) ** 2)
            return func

        def g_4(i, n):
            t4 = 0
            for x in range(i, n):
                t4 = t4 + (c[x][3] * delta(x) ** c[x][1] * self.delta * psi(x))
            return t4

        term_1 = g_1(0, 7)
        term_2 = g_2(7, 51)
        term_3 = g_3(51, 54)
        term_4 = g_4(0, 2)

        phi_r = term_1 + term_2 + term_3 + term_4

        return phi_r

    def phi_r_der_del(self):
        a = np.array([[0.0, 1, -0.5, 0.12533547935523e-1],
                      [0.0, 1, 0.875, 0.78957634722828e1],
                      [0.0, 1, 1, -0.87803203303561e1],
                      [0.0, 2, 0.5, 0.31802509345418],
                      [0.0, 2, 0.75, -0.26145533859358],
                      [0.0, 3, 0.375, -0.78199751687981e-2],
                      [0.0, 4, 1, 0.88089493102134e-2],
                      [1, 1, 4, -0.66856572307965],
                      [1, 1, 6, 0.20433810950965],
                      [1, 1, 12, -0.66212605039687e-4],
                      [1, 2, 1, -0.19232721156002],
                      [1, 2, 5, -0.25709043003438],
                      [1, 3, 4, 0.16074868486251],
                      [1, 4, 2, -0.40092828925807e-1],
                      [1, 4, 13, 0.39343422603254e-6],
                      [1, 5, 9, -0.75941377088144e-5],
                      [1, 7, 3, 0.56250979351888e-3],
                      [1, 9, 4, -0.15608652257135e-4],
                      [1, 10, 11, 0.11537996422951e-8],
                      [1, 11, 4, 0.36582165144204e-6],
                      [1, 13, 13, -0.13251180074668e-11],
                      [1, 15, 1, -0.62639586912454e-9],
                      [2, 1, 7, -0.10793600908932],
                      [2, 2, 1, 0.17611491008752e-1],
                      [2, 2, 9, 0.22132295167546],
                      [2, 2, 10, -0.40247669763528],
                      [2, 3, 10, 0.58083399985759],
                      [2, 4, 3, 0.49969146990806e-2],
                      [2, 4, 7, -0.31358700712549e-1],
                      [2, 4, 10, -0.74315929710341],
                      [2, 5, 10, 0.47807329915480],
                      [2, 6, 6, 0.20527940895948e-1],
                      [2, 6, 10, -0.13636435110343],
                      [2, 7, 10, 0.14180634400617e-1],
                      [2, 9, 1, 0.83326504880713e-2],
                      [2, 9, 2, -0.29052336009585e-1],
                      [2, 9, 3, 0.38615085574206e-1],
                      [2, 9, 4, -0.20393486513704e-1],
                      [2, 9, 8, -0.16554050063734e-2],
                      [2, 10, 6, 0.19955571979541e-2],
                      [2, 10, 9, 0.15870308324157e-3],
                      [2, 12, 8, -0.16388568342530e-4],
                      [3, 3, 16, 0.43613615723811e-1],
                      [3, 4, 22, 0.34994005463765e-1],
                      [3, 4, 23, -0.76788197844621e-1],
                      [3, 5, 23, 0.22446277332006e-1],
                      [4, 14, 10, -0.62689710414685e-4],
                      [6, 3, 50, -0.55711118565645e-9],
                      [6, 6, 44, -0.19905718354408],
                      [6, 6, 46, 0.31777497330738],
                      [6, 6, 50, -0.11841182425981],
                      [0.0, 3, 0, -0.31306260323435e2, 20, 150, 1.21, 1],
                      [0.0, 3, 1, 0.31546140237781e2, 20, 150, 1.21, 1],
                      [0.0, 3, 4, -0.25213154341695e4, 20, 250, 1.25, 1]], dtype=object)

        def f(x):
            func = a[x][3] * self.delta ** (a[x][1] - 1) * self.tau ** a[x][2]
            return func

        def h(x):
            func = np.exp(-a[x][4] * (self.delta - a[x][7]) ** 2 - a[x][5] * (self.tau - a[x][6]) ** 2)
            return func

        def g_1(i, n):
            t1 = 0
            for x in range(i, n):
                t1 = t1 + a[x][1] * f(x)
            return t1

        def g_2(i, n):
            t2 = 0
            for x in range(i, n):
                t2 = t2 + f(x) * np.exp(-self.delta ** a[x][0]) * (a[x][1] - a[x][0] * self.delta ** a[x][0])
            return t2

        def g_3(i, n):
            t3 = 0
            for x in range(i, n):
                t3 = t3 + (self.delta * f(x) * h(x)) * ((a[x][1] / self.delta) - 2 * a[x][4] * (self.delta - a[x][7]))
            return t3

        c = np.array([[3.5, 0.85, 0.2, -0.14874640856724, 28, 700, 0.32, 0.3],
                      [3.5, 0.95, 0.2, 0.31806110878444, 32, 800, 0.32, 0.3]])

        def theta(x):
            func = (1 - self.tau) + c[x][6] * ((self.delta - 1) ** 2) ** (1 / (2 * c[x][7]))
            return func

        def delta(x):
            func = theta(x) ** 2 + c[x][2] * ((self.delta - 1) ** 2) ** c[x][0]
            return func

        def delta_bi_der_del(x):
            func = c[x][1] * delta(x) ** (c[x][1] - 1) * del_der_del(x)
            return func

        def del_der_del(x):
            func = (self.delta - 1) * ((c[x][6] * theta(x) * (2 / c[x][7]) * ((self.delta - 1) ** 2) ** ((1 / (2 * c[x][7])) - 1)) + (2 * c[x][2] * c[x][0] * ((self.delta - 1) ** 2) ** (c[x][0] - 1)))
            return func

        def psi(x):
            func = np.exp(-c[x][4] * (self.delta - 1) ** 2 - c[x][5] * (self.tau - 1) ** 2)
            return func

        def psi_der_del(x):
            func = -2 * c[x][4] * (self.delta - 1) * psi(x)
            return func

        def g_4(i, n):
            t4 = 0
            for x in range(i, n):
                t4 = t4 + (c[x][3] * (delta(x) ** c[x][1] * (psi(x) + self.delta * psi_der_del(x)) + delta_bi_der_del(x) * self.delta * psi(x)))
            return t4

        term_1 = g_1(0, 7)
        term_2 = g_2(7, 51)
        term_3 = g_3(51, 54)
        term_4 = g_4(0, 2)

        phi_r_der_del = term_1 + term_2 + term_3 + term_4

        return phi_r_der_del

    def density_brentq(self):

        # pressure from Pa to MPa
        p = self.p / 1000000

        # boundaries
        scalar_1 = 6.37e-6 + -4.01e-6 * np.log(p)
        scalar_2 = -9.22e-3 + 4.16e-3 * np.log(p)
        scalar_3 = 3.53 + -1.43 * np.log(p)
        scalar_4 = 597 + 163 * np.log(p)

        bound_func = scalar_1 * self.t ** 3 + scalar_2 * self.t ** 2 + scalar_3 * self.t + scalar_4
        scope = 15

        a = bound_func - scope
        b = bound_func + scope

        # brentq root finding method
        def f(x):
            term_1 = WaterWagner(self.t, self.p, x).phi_r_der_del() * ((x ** 2) / self.dc)
            term_2 = x
            term_3 = self.p / (self.R * 1000 * self.t)
            func = term_1 + term_2 - term_3
            return func

        density = optimize.brentq(f, a, b)

        return density

    def density_fsolve(self):
        a = 1
        b = 2000

        scope = np.array([600, 1200])

        def f(x):
            term_1 = WaterWagner(self.t, self.p, x).phi_r_der_del() * ((x ** 2) / self.dc)
            term_2 = x
            term_3 = self.p / (self.R * 1000 * self.t)
            func = term_1 + term_2 - term_3
            return func

        density = optimize.fsolve(f, scope)

        return density[-1]

    def free_energy(self):
        # Helmholtz free energy

        ideal_gas = self.phi_naught()
        residual = self.phi_r()

        helmholtz_free_energy = ideal_gas + residual

        return helmholtz_free_energy
