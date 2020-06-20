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
        res_val = np.array([[0.999984195, -2.718e-08], [0.9999840270, -2.398e-08], [0.999983827, -2.049e-08]])
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
        res_val[:, 0] = [0.5575667595709363, 0.5327362189194406, 0.49330648204453487, 0.4262814916316443,
                         0.39285383548929875, 0.38292354454074434]
        res_val[:, 1] = [-1.5543122344752192e-15, -4.163336342344337e-17, 4.163336342344337e-17,
                         -2.6020852139652106e-16, - 3.4867941867133823e-16, -6.546586189815073e-14]
        c_phi = np.zeros([v_mat.shape[0], 2])
        for ind, u_val in enumerate(v_mat):
            phi_s = b_sph.phi(u_val)
            eqn = b_sph.eqn_min_phi(u_val, phi_s)
            c_phi[ind, 0] = phi_s
            c_phi[ind, 1] = eqn

        t_mat = [2e-16, 2.0e-16]
        for ind in range(2):
            exp1 = np.allclose(res_val[:, ind], c_phi[:, ind], atol=t_mat[ind], rtol=0.0)
            self.assertTrue(exp1)

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
        cp_vals = np.array([h_val, fr[0]])
        ac_vals = np.array([11.750745890704016, 145.65149475350745])
        exp = np.allclose(cp_vals, ac_vals, atol=1e-13)
        self.assertTrue(exp)

        # compute optimal brush size
        res = b_sph.optimal_lambda()
        res_comp = 0.03264222743670685 - b_sph.chi
        fun_comp = 145.06444193602587
        res_diff = np.abs(res_comp - res.x)
        fun_diff = np.abs(fun_comp-res.fun)
        self.assertLess(res_diff, 1e-8)
        self.assertLess(fun_diff, 1e-13)

        # compare free energies, brush size, optimal Lambda and free energy derivative
        delta_res = 0.97 * np.abs(res.x+b_sph.chi)
        vals = res.x + np.linspace(-delta_res, delta_res, 11)
        free_energy = np.zeros_like(vals)
        h_vals = np.zeros_like(vals)
        d_vals = np.zeros_like(vals)
        f_comp = np.array([145.65211862, 145.47157141, 145.31436453, 145.18690686, 145.09864659,
                           145.06444194, 145.10980527, 145.28395308, 145.69043856, 146.46990222, 147.57747715])
        h_comp = np.array([11.75017423, 11.93516383, 12.14449686, 12.38515022, 12.66761464, 13.00871698, 13.43784993,
                           14.01204153, 14.84611594, 16.05257333, 17.48202434])
        l_comp = np.array([0.00097927, 0.00731186, 0.01364445, 0.01997704, 0.02630964, 0.03264223, 0.03897482,
                           0.04530741, 0.05164, 0.0579726, 0.06430519])
        d_comp = np.array([-1.0916966246944142, -0.8652269502955683, -0.6422216779856562, -0.4231721404736284,
                           -0.20876074766049868, -1.0557409524381001e-07, 0.20145925688545674, 0.3925406913593079,
                           0.5666507519715793, 0.7131280432656801, 0.8312009645838874])
        for ind, lam_lg in enumerate(vals):
            fe_b = Bb.BinaryBrush(dim, chi, sigma, rad, pl_peo, lam_lg)
            h_calc = fe_b.determine_h()
            # brush size
            h_vals[ind] = h_calc
            # free energy
            free_energy[ind] = fe_b.free_energy()[0]
            # free energy derivative
            d_vals[ind] = fe_b.der_free_energy()

        self.assertTrue(np.allclose(f_comp, free_energy))
        self.assertTrue(np.allclose(h_vals, h_comp))
        self.assertTrue(np.allclose(l_comp-chi, vals))
        self.assertTrue(np.allclose(d_comp, d_vals, atol=1e-6, rtol=0.0))


if __name__ == '__main__':
    unittest.main()
