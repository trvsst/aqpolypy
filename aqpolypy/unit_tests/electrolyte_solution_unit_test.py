"""
:module: electrolyte_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for electrolyte solution

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, April2023
.. history:
..
"""
import numpy as np
import unittest
import aqpolypy.free_energy_polymer.ElectrolyteSolution as El


class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        self.temp = 300
        self.param_w = {'v_w': 30, 'de_w': 1330, 'se_w': 3.62, 'de_2d': 0.0, 'ds_2d': 0.0, 'de_2a': 0.0, 'ds_2a': 0.0}
        dict_vol = {'v_s': 30, 'v_b': 35}
        dict_p = {'de_p0': 665, 'ds_p0': 0.0, 'de_p1': -540, 'ds_p1': 0.0, 'de_p2': -10000, 'ds_p2': 0.0}
        dict_bp = {'de_bp0': 834, 'ds_bp0': 0.0, 'de_bp1': -658.0, 'ds_bp1': 0.0, 'de_bp2': -10000, 'ds_bp2': 0.0}
        dict_m = {'de_m0': 882, 'ds_m0': 0.0, 'de_m1': -263, 'ds_m1': 0.0, 'de_m2': -10000, 'ds_m2': 0.0}
        dict_bm = {'de_bm0': 1079, 'ds_bm0': 0.0, 'de_bm1': -248, 'ds_bm1': 0.0, 'de_bm2': -10000, 'ds_bm2': 0.0}
        dict_b = {'de_b': np.log(0.7), 'ds_b': 0.0}
        self.param_salt = {**dict_vol, **dict_p, **dict_bp, **dict_m, **dict_bm, **dict_b}
        dict_max = {'m_p': 9.77, 'm_m': 8.21, 'mb_p': 5.5, 'mb_m': 5.9}
        dict_hp = {'h_p0': 5.0, 'h_p1': 1.0, 'h_p2': 0.0}
        dict_h_m = {'h_m0': 6.0, 'h_m1': 1.5, 'h_m2': 0.0}
        dict_h_bp = {'hb_p0': 4.0, 'hb_p1': 0.5, 'hb_p2': 0.0}
        dict_h_bm = {'hb_m0': 4.5, 'hb_m1': 1.0, 'hb_m2': 0.0}
        self.param_h = {**dict_max, **dict_hp, **dict_h_m, **dict_h_bp, **dict_h_bm}

    def test_free_ideal(self):

        el_sol = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)

        test_n_w = np.array([55.54, 55.0, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0])
        comp_ideal = el_sol.f_ideal(test_n_w, test_n_s)
        vals_ideal = np.array([10694.17289883, 10625.31694141, 11800.63110599])

        # testing params up to a precision of 10^-6
        test_i = np.allclose(comp_ideal, vals_ideal, 0, 1e-6)
        self.assertTrue(test_i)

    def test_free_compressibilty(self):

        el_sol = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)
        test_n_w = np.array([55.54, 55.0, 30.5, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0, 20.5])
        test_f_b = np.array([0.0, 0.0, 0.5, 1])

        k_ref = 5e-3
        comp_comp = el_sol.f_comp(test_n_w, test_n_s, test_f_b, k_ref)
        vals_comp = np.array([16642.00600168, 16630.00600601, 15630.00638978, 16305.00612557])
        test_c = np.allclose(comp_comp, vals_comp, 0, 1e-6)
        self.assertTrue(test_c)

    def test_free_debye_huckel(self):

        el_sol = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_bpar = np.array([1e-2, 1e-2, 0.5, 1.0])

        comp_debye = el_sol.f_debye(test_n_w, test_n_s, test_f_b, test_bpar)
        vals_comp = [-0.00156919, -0.55497243, -0.77886876, -0.62931197]
        test_d = np.allclose(comp_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_d)

    def test_free_assoc(self):

        el_sol = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        comp_assoc = el_sol.f_assoc(test_n_w, test_n_s, test_y, test_za, test_zd, test_f_b)
        vals_comp = [-17928.98280199, -17246.08272997, -22121.72593814, -23873.49131049]
        test_a = np.allclose(comp_assoc, vals_comp, 0, 1e-6)
        self.assertTrue(test_a)

    def test_mu_water_1(self):
        el_mu = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        comp_mu_w_1=el_mu.mu_w_1(test_n_w, test_n_s, test_y, test_za, test_zd, test_f_b)
        vals_comp = [-5.25727997, -4.55696617, -7.17320148, -7.61640162]
        test_mu_1 = np.allclose(comp_mu_w_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_1)

    def test_mu_water_debye(self):
        el_mu_d = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_bpar = np.array([1e-2, 1e-2, 0.5, 1.0])

        test_m = el_mu_d.concentration_molal(test_n_w, test_n_s)
        comp_mu_debye = el_mu_d.mu_w_debye(test_m, test_f_b, test_bpar)
        vals_comp = [1.41161015e-05, 5.01848558e-03, 5.43015396e-03, 3.57547593e-03]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_comp(self):
        el_mu_c = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)

        test_n_w = np.array([55.54, 55.0, 30.5, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0, 20.5])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_f_b = np.array([0.0, 0.0, 0.5, 1])

        comp_mu_c = el_mu_c.mu_w_comp(test_n_w, test_n_s, test_y, test_f_b)
        vals_comp = [ 333.24,  337.5,  6516.0,  5937.6 ]
        test_mu_comp = np.allclose(comp_mu_c, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_comp)

    def test_mu_salt_1(self):
        el_mu = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        comp_mu_salt_1 = el_mu.mu_sf_1(test_n_w, test_n_s, test_y, test_za, test_zd, test_f_b)
        vals_comp = [45.40514735, 53.53755282, 58.10210904, 64.73013653]
        test_mu_salt_1 = np.allclose(comp_mu_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_salt_1)

    def test_mu_salt_debye(self):
        el_mu_d = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_bpar = np.array([1e-2, 1e-2, 0.5, 1.0])

        test_m = el_mu_d.concentration_molal(test_n_w, test_n_s)
        comp_mu_debye = el_mu_d.mu_w_debye(test_m, test_f_b, test_bpar)
        vals_comp = [1.41161015e-05, 5.01848558e-03, 5.43015396e-03, 3.57547593e-03]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_salt_bjerrum_1(self):
        el_mu = El.ElectrolyteSolution(self.temp, self.param_w, self.param_salt, self.param_h)

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_f_b = np.array([1e-4, 1e-4, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        comp_mu_b_salt_1 = el_mu.mu_sb_1(test_n_w, test_n_s, test_y, test_za, test_zd, test_f_b)
        vals_comp = [-26.49673298, -22.36947333, -11.31812004,  -5.83708178]
        test_mu_b_salt_1 = np.allclose(comp_mu_b_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_b_salt_1)


if __name__ == '__main__':
    unittest.main()
