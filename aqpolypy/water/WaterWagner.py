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


class WaterWagner:

    def __init__(self, d, t):
        self.d = d
        self.t = t

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

    def phi_tau(self):
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
                      [6, 6, 50, -0.11841182425981]])

        def f_1(x):
            func = a[x][3] * self.delta ** a[x][1] * self.tau ** a[x][2]
            return func

        def g_1(i, n):
            t1 = 0
            for x in range(i, n):
                t1 = t1 + f_1(x)
            return t1

        def f_2(x):
            func = a[x][3] * self.delta ** a[x][1] * self.tau ** a[x][2] * np.exp(-self.delta ** a[x][0])
            return func

        def g_2(i, n):
            t2 = 0
            for x in range(i, n):
                t2 = t2 + f_2(x)
            return t2

        b = np.array([[0.0, 3, 0, -0.31306260323435e2, 20, 150, 1.21, 1],
                      [0.0, 3, 1, 0.31546140237781e2, 20, 150, 1.21, 1],
                      [0.0, 3, 4, -0.25213154341695e4, 20, 250, 1.25, 1]])

        def f_3(x):
            func = b[x][3] * self.delta ** b[x][1] * self.tau ** b[x][2]
            return func

        def h_3(x):
            func = np.exp(-b[x][4] * (self.delta - b[x][7]) ** 2 - b[x][5] * (self.tau - b[x][6]) ** 2)
            return func

        def g_3(i, n):
            t3 = 0
            for x in range(i, n):
                t3 = t3 + (f_3(x) * h_3(x))
            return t3

        c = np.array([[3.5, 0.85, 0.2, -0.14874640856724, 28, 700, 0.32, 0.3],
                      [3.5, 0.95, 0.2, 0.31806110878444, 32, 800, 0.32, 0.3]])

        def f_4(x):
            func = ((1 - self.tau) + c[x][6] * ((self.delta - 1) ** 2) ** (1 / (2 * c[x][7]))) ** 2
            return func

        def h_4(x):
            func = c[x][2] * ((self.delta - 1) ** 2) ** c[x][0]
            return func

        def j_4(x):
            func = np.exp(-c[x][4] * (self.delta - 1) ** 2 - c[x][5] * (self.tau - 1) ** 2)
            return func

        def g_4(i, n):
            t4 = 0
            for x in range(i, n):
                t4 = t4 + (c[x][3] * (f_4(x) + h_4(x)) ** c[x][1] * self.delta * j_4(x))
            return t4

        term_1 = g_1(0, 7)
        term_2 = g_2(7, 51)
        term_3 = g_3(0, 3)
        term_4 = g_4(0, 2)

        phi_tau = term_1 + term_2 + term_3 + term_4

        return phi_tau

    def free_energy(self):
        # Helmholtz free energy

        ideal_gas = self.phi_naught()
        residual = self.phi_tau()

        helmholtz_free_energy = ideal_gas + residual

        return helmholtz_free_energy
