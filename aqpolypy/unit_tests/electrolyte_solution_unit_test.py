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
        self.in_p = np.zeros([16, 4])
        self.in_p[3, :] = dict_hp['h_p0']
        self.in_p[4, :] = dict_hp['h_p1']
        self.in_p[5, :] = dict_hp['h_p2']
        self.in_p[6, :] = dict_h_m['h_m0']
        self.in_p[7, :] = dict_h_m['h_m1']
        self.in_p[8, :] = dict_h_m['h_m2']
        self.in_p[9, :] = dict_h_bp['hb_p0']
        self.in_p[10, :] = dict_h_bp['hb_p1']
        self.in_p[11, :] = dict_h_bp['hb_p2']
        self.in_p[12, :] = dict_h_bm['hb_m0']
        self.in_p[13, :] = dict_h_bm['hb_m1']
        self.in_p[14, :] = dict_h_bm['hb_m2']

    def test_free_ideal(self):
        test_n_w = np.array([55.54, 55.0, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0])

        el_sol = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h)

        comp_ideal = el_sol.f_ideal()
        vals_ideal = np.array([10694.17289883, 10625.31694141, 11800.63110599])

        # testing params up to a precision of 10^-6
        test_i = np.allclose(comp_ideal, vals_ideal, 0, 1e-6)
        self.assertTrue(test_i)

    def test_free_compressibilty(self):
        k_ref = 5e-3
        test_n_w = np.array([55.54, 55.0, 30.5, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0, 20.5])

        el_sol = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h, k_r= k_ref)

        self.in_p[15] = np.array([0.0, 0.0, 0.5, 1])


        comp_comp = el_sol.f_comp(self.in_p)
        vals_comp = np.array([16642.00600168, 16630.00600601, 15630.00638978, 16305.00612557])
        test_c = np.allclose(comp_comp, vals_comp, 0, 1e-6)
        self.assertTrue(test_c)

    def test_free_debye_huckel(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        test_bpar = np.array([1e-2, 1e-2, 0.5, 1.0])

        el_sol = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h, b_param=test_bpar)

        self.in_p[15] = np.array([0.0, 0.0, 0.25, 0.25])

        comp_debye = el_sol.f_debye(self.in_p)
        vals_comp = [-0.04707576, -16.64917299, -23.36606274, -18.8793592]
        test_d = np.allclose(comp_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_d)

    def test_free_assoc(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        el_sol = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        self.in_p[0] = test_y
        self.in_p[1] = test_za
        self.in_p[2] = test_zd
        self.in_p[15] = test_f_b

        comp_assoc = el_sol.f_assoc(self.in_p)
        vals_comp = [-17928.98280199, -17246.08272997, -22121.72593814, -23873.49131049]
        test_a = np.allclose(comp_assoc, vals_comp, 0, 1e-6)
        self.assertTrue(test_a)

    def test_mu_water_1(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        el_mu = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        self.in_p[0] = test_y
        self.in_p[1] = test_za
        self.in_p[2] = test_zd
        self.in_p[15] = test_f_b
        comp_mu_w_1=el_mu.mu_w_1(self.in_p)
        vals_comp = [-5.25727997, -4.55696617, -7.17320148, -7.61640162]
        test_mu_1 = np.allclose(comp_mu_w_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_1)

    def test_mu_water_debye(self):
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        t_b = np.array([1e-2, 1e-2, 0.5, 1.0])

        el = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h, b_param=t_b)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        self.in_p[15] = test_f_b

        comp_mu_debye = el.mu_w_debye(self.in_p)
        vals_comp = [1.41161015e-05, 5.01848558e-03, 5.43015396e-03, 3.57547593e-03]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_comp(self):

        test_n_w = np.array([55.54, 55.0, 30.5, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0, 20.5])

        el = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h, press=0)

        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_f_b = np.array([0.0, 0.0, 0.5, 1])

        self.in_p[0] = test_y
        self.in_p[15] = test_f_b

        comp_mu_c = el.mu_w_comp(self.in_p)
        vals_comp = [ 333.24,  337.5,  6516.0,  5937.6 ]
        test_mu_comp = np.allclose(comp_mu_c, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_comp)

    def test_mu_salt_1(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        el_mu = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        self.in_p[0] = test_y
        self.in_p[1] = test_za
        self.in_p[2] = test_zd
        self.in_p[15] = test_f_b

        comp_mu_salt_1 = el_mu.mu_sf_1(self.in_p)
        vals_comp = [45.86428195, 53.99668742, 58.56124364, 65.18927112]
        test_mu_salt_1 = np.allclose(comp_mu_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_salt_1)

    def test_mu_salt_debye(self):
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])
        t_b = np.array([1e-2, 1e-2, 0.5, 1.0])

        el = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h, b_param=t_b)

        self.in_p[15] = np.array([0.0, 0.0, 0.25, 0.25])

        comp_mu_debye = el.mu_sf_debye(self.in_p)
        vals_comp = [-0.23532007, -1.66197828, -1.43308287, -1.09890055]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_salt_bjerrum_1(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        el_mu = El.ElectrolyteSolution(test_n_w, test_n_s, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([1e-4, 1e-4, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        self.in_p[0] = test_y
        self.in_p[1] = test_za
        self.in_p[2] = test_zd
        self.in_p[15] = test_f_b

        comp_mu_b_salt_1 = el_mu.mu_sb_1(self.in_p)
        vals_comp = [-26.5958937, -22.46863405, -11.41728076, -5.9362425]
        test_mu_b_salt_1 = np.allclose(comp_mu_b_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_b_salt_1)

    def test_sols(self):

        def sol_water(df):
            y_val = 1+0.25*np.exp(-df)*(1-np.sqrt(1+8*np.exp(df)))
            return np.array([y_val, y_val**2, y_val**2])

        nw = 55.5
        ns = 1e-14

        ini_p = np.zeros(16)

        tp = np.array([300, 320, 340, 360, 380])

        for temp in tp:
            el = El.ElectrolyteSolution(nw, ns, temp, self.param_w, self.param_salt, self.param_h)
            ini_p[0] = 0.63
            ini_p[1] = 0.4
            ini_p[2] = 0.4
            ini_p[15] = 1e-14
            sol = el.solve_eqns(ini_p, np.array([0,1,2]))
            test_cond = np.allclose(sol, sol_water(el.f_w), 0, 1e-8)
            self.assertTrue(test_cond)


if __name__ == '__main__':
    unittest.main()
