"""
:module: polymer_brush_spherical_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, June2020
.. history:
"""
import unittest
import numpy as np

import aqpolypy.polymer.PEOSimple as PeO
import aqpolypy.free_energy_polymer.BinaryBrush as Bb


class TestPolymerFreeSpherical(unittest.TestCase):

    def test_read_params(self):
        """checks that the parameters are read correctly"""

        dim = 3
        xi_t = 5.0
        chi = 0.5 * (1 / xi_t + 1)
        sigma = 1.9
        rad = 0.5 * 90.72
        lag = 1e-3 - chi

        # let us take the PEO with molecular weight 5000
        pl_peo = PeO.PEOSimple(5000)

        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag)

        xi_t = b_sph.xi_t
        xi_s = b_sph.xi_s
        n_p = pl_peo.n_p
        hat_r = b_sph.hat_r
        r_vol = b_sph.r_vol
        n_chains = b_sph.num_chains
        res_calc = np.array([xi_t, xi_s, n_p, hat_r, r_vol])
        res_correct = np.array([5, 1.7158202388509698, 57.06465437882804, 6.265193370165745, 4.327379464250886])
        vals = np.allclose(res_calc, res_correct)
        self.assertTrue(vals)
        n_chains_correct = 7427.82
        self.assertTrue(np.abs(n_chains-n_chains_correct) < 1e-2)

    def test_phi(self):
        """Testing function phi, eqn_min_phi"""
        # tests the calculation of phi

        # let us take the PEO with molecular weight 5000
        pl_peo = PeO.PEOSimple(5000)

        # sphere test 1
        dim = 3
        xi_t = 0.1
        chi = 0.5*(1/xi_t+1)
        sigma = 1.9
        rad = 0.5*90.72
        lag = 1e-3-chi

        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag)

        u_mat = np.array([0.0, 0.5, 1.5])
        res_val = np.array([[0.999983934, -2.3895e-08], [0.999983763, -2.0928e-08], [0.999983560, -1.771e-08]])
        comp_phi = np.zeros([u_mat.shape[0], 2])
        for ind, u_val in enumerate(u_mat):
            phi_s = b_sph.phi(u_val)
            eqn = b_sph.eqn_min_phi(u_val, phi_s)
            comp_phi[ind, 0] = phi_s
            comp_phi[ind, 1] = eqn

        t_mat = [1e-9, 1e-11]
        for ind in range(2):
            exp1 = np.allclose(res_val[:, ind], comp_phi[:, ind], atol=t_mat[ind], rtol=0.0)
            self.assertTrue(exp1)

        # sphere test2
        xi_t = 5.0
        chi = 0.5 * (1 / xi_t + 1)
        lag = 1e-3 - chi

        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag)

        v_mat = np.array([0.0, 0.5, 1.5, 4.5, 8.5, 11.75])
        res_val = np.zeros([v_mat.shape[0], 2])
        res_val[:, 0] = [0.54512457, 0.51832402, 0.47476192, 0.39479913, 0.34697209, 0.32967486]
        res_val[:, 1] = [-5.55111512e-17,  0.00000000e+00, -2.77555756e-17,  5.20417043e-17, -1.29445066e-14,
                         -1.14491749e-16]
        c_phi = np.zeros([v_mat.shape[0], 2])
        for ind, u_val in enumerate(v_mat):
            phi_s = b_sph.phi(u_val)
            eqn = b_sph.eqn_min_phi(u_val, phi_s)
            c_phi[ind, 0] = phi_s
            c_phi[ind, 1] = eqn

        t_mat = [2e-8, 2.0e-16]
        for ind in range(2):
            exp1 = np.allclose(res_val[:, ind], c_phi[:, ind], atol=t_mat[ind], rtol=0.0)
            self.assertTrue(exp1)

        # check normalization
        norm = b_sph.phi_normalization()
        self.assertTrue(np.abs(norm[0]-1) < 1e-10)

        # check inverse function
        u_val = b_sph.inv_phi(c_phi[:, 0])
        self.assertTrue(np.allclose(u_val, v_mat))

    def test_free(self):
        """Testing free energy and associated functions"""

        # let us take the PEO with molecular weight 5000
        pl_peo = PeO.PEOSimple(5000)

        # sphere test 1
        dim = 3
        # sphere test2
        xi_t = 5.0
        chi = 0.5 * (1 / xi_t + 1)
        sigma = 1.9
        rad = 0.5 * 90.72
        lag = 1e-3 - chi

        # compare brush size and free energy for a given value of Lambda
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag)
        h_val = b_sph.determine_h()
        fr = b_sph.free_energy()
        cp_vals = np.array([h_val, fr[0][0], fr[1][0], fr[2][0]])
        ac_vals = np.array([12.4543068094379, 52.25261599, 93.88085416, 5.08157019])
        exp = np.allclose(cp_vals, ac_vals, atol=1e-13)
        self.assertTrue(exp)

        # compute optimal brush size
        res = b_sph.optimal_lambda()
        res_comp = -0.5929852656156657
        fun_comp = 151.18609253170362
        res_diff = np.abs(res_comp - res.x)
        fun_diff = np.abs(fun_comp-res.fun)
        self.assertLess(res_diff, 1e-8)
        self.assertLess(fun_diff, 1e-13)

        # compare free energies, brush size, optimal Lambda and free energy derivative
        delta_res = 0.97 * np.abs(res.x+b_sph.chi)
        vals = res.x + np.linspace(-delta_res, delta_res, 11)
        free_energy = np.zeros([vals.shape[0], 3])
        h_vals = np.zeros_like(vals)
        d_vals = np.zeros_like(vals)
        f_comp = np.array([[52.57338855, 93.5907969, 5.05843421], [52.01845303, 94.09298618, 5.09861057],
                           [51.45344121, 94.60620618, 5.14026642], [50.87778347, 95.13109749, 5.18351718],
                           [50.29085432, 95.6683648, 5.2284931 ], [49.69196466, 96.21878595, 5.27534192],
                           [49.08035284, 96.78322257, 5.32423227], [48.45517411, 97.36263262, 5.37535786],
                           [47.81548843, 97.95808509, 5.42894285], [47.16024629, 98.5707773, 5.48524865],
                           [46.48827252, 99.20205492, 5.54458278]])
        h_comp = np.array([12.41978285, 12.47976726, 12.54212312, 12.60704343, 12.67474614, 12.74547861, 12.81952321,
                           12.89720425, 12.97889664, 13.06503691, 13.15613726])
        l_comp = np.array([-0.59978956, -0.5984287,  -0.59706784, -0.59570698, -0.59434612, -0.59298527, -0.59162441,
                           -0.59026355, -0.58890269, -0.58754183, -0.58618097])
        d_comp = np.array([-2.33270867e-01, -1.86167508e-01, -1.39278773e-01, -9.26141030e-02, -4.61839588e-02,
                           6.57251675e-09, 4.59247282e-02, 9.15754564e-02, 1.36935411e-01,
                           1.81985337e-01, 2.26702928e-01])

        for ind, lam_lg in enumerate(vals):
            fe_b = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lam_lg)
            h_calc = fe_b.determine_h()
            # brush size
            h_vals[ind] = h_calc
            # free energy
            fr = fe_b.free_energy()
            for ind_p in range(3):
                free_energy[ind, ind_p] = fr[ind_p][0]
            # free energy derivative
            d_vals[ind] = fe_b.der_free_energy()

        self.assertTrue(np.allclose(h_vals, h_comp))
        self.assertTrue(np.allclose(l_comp, vals))
        self.assertTrue(np.allclose(d_comp, d_vals, atol=1e-6, rtol=0.0))
        self.assertTrue(np.allclose(f_comp, free_energy))

    def test_solveh(self):
        """testing solving for the lagrange parameter that gives a brush size"""

        # let us take the PEO with molecular weight 5000
        pl_peo = PeO.PEOSimple(5000)

        # sphere test 1
        dim = 3
        # sphere test2
        xi_t = 5.0
        chi = 0.5 * (1 / xi_t + 1)
        sigma = 1.9
        rad = 0.5 * 90.72
        lag_ini = 1e-3 - chi

        # compare brush size and free energy for a given value of Lambda
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        h_val = b_sph.determine_h()*pl_peo.k_length
        c_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        c_sph.determine_lagrange(h_val)
        self.assertLess(np.abs(c_sph.lag-b_sph.lag), 1e-8)

        # starting from a different value of the lagrange parameter
        lag_ini = 0.1
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        h_val = b_sph.determine_h() * pl_peo.k_length
        lag_ini_t = 1
        c_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini_t)
        c_sph.determine_lagrange(h_val)
        self.assertLess(np.abs(c_sph.lag - b_sph.lag), 1e-7)

        # different values of chi=1.0
        xi_t = 1.0
        chi = 0.5 * (1 / xi_t + 1)
        lag_ini = 1e-3 - chi
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        h_val = b_sph.determine_h() * pl_peo.k_length
        c_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        c_sph.determine_lagrange(h_val)
        self.assertLess(np.abs(c_sph.lag - b_sph.lag), 1e-7)

        # different values of chi=0.5
        xi_t = 0.5
        chi = 0.5 * (1 / xi_t + 1)
        lag_ini = 1e-1 - chi
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        h_val = b_sph.determine_h() * pl_peo.k_length
        c_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        c_sph.determine_lagrange(h_val)
        self.assertLess(np.abs(c_sph.lag - b_sph.lag), 1e-7)

        # different values of chi=0.1, optimal value of h
        xi_t = 0.1
        chi = 0.5 * (1 / xi_t + 1)
        lag_ini = 1e-3 - chi
        b_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        res = b_sph.optimal_lambda()
        h_val = b_sph.determine_h() * pl_peo.k_length
        c_sph = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lag_ini)
        c_sph.determine_lagrange(h_val, lag=-1.6)
        self.assertLess(np.abs(c_sph.lag - b_sph.lag), 1e-7)


if __name__ == '__main__':
    unittest.main()
