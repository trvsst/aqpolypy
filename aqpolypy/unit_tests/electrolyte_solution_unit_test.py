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
import aqpolypy.units.units as un

class TestFreeEnergy(unittest.TestCase):

    def setUp(self):
        # define dict
        self.temp = 300
        self.delta_w = un.delta_w()
        self.param_w = {'v_w': 30, 'de_w': 1330, 'ds_w': 3.62, 'de_2d': 0.0, 'ds_2d': 0.0, 'de_2a': 0.0, 'ds_2a': 0.0}
        dict_vol = {'v_s': 30, 'v_b': 30}
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
        self.param_h = {**dict_max}
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

        test_m = self.delta_w*test_n_s/test_n_w

        el_sol = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h)

        comp_ideal = el_sol.f_ideal()
        vals_ideal = np.array([[-1.0,-1.10283371,-2.4342409]])
        # testing params up to a precision of 10^-6
        test_i = np.allclose(comp_ideal, vals_ideal, 0, 1e-6)
        self.assertTrue(test_i)

    def test_free_compressibilty(self):
        k_ref = 5e-3

        test_m = np.array([1e-3, 1e-2, 1e-1, 1])
        param_salt = self.param_salt
        param_salt['v_b'] = 35
        el_sol = El.ElectrolyteSolution(test_m, self.temp, self.param_w, param_salt, self.param_h, k_r= k_ref)

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[15] = np.array([0.0, 0.0, 0.5, 1])

        comp_comp = el_sol.f_comp(in_p)
        vals_comp = np.array([0.00000000e+00, 1.23259516e-31, 2.24538339e-07, 8.67343930e-05])
        test_c = np.allclose(comp_comp, vals_comp, 0, 1e-6)
        self.assertTrue(test_c)

    def test_free_debye_huckel(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        test_bpar = np.array([1e-2, 1e-2, 0.5, 1.0])

        el_sol = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h, b_param=test_bpar)

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[15] = np.array([0.0, 0.0, 0.25, 0.25])

        comp_debye = el_sol.f_debye(in_p)
        vals_comp = [-2.82482823e-05, -9.99950330e-03, -1.40336713e-02, -1.13389545e-02]
        test_d = np.allclose(comp_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_d)

    def test_free_assoc(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        el_sol = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[0] = test_y
        in_p[1] = test_za
        in_p[2] = test_zd
        in_p[15] = test_f_b

        comp_assoc = el_sol.f_assoc(in_p)
        vals_comp = [-2.57388926, -2.72540227, -2.73953532, -2.72985608]
        test_a = np.allclose(comp_assoc, vals_comp, 0, 1e-6)
        self.assertTrue(test_a)

    def test_mu_water_ideal_assoc(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        el_mu = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[0] = test_y
        in_p[1] = test_za
        in_p[2] = test_zd
        in_p[15] = test_f_b
        comp_mu_w_1=el_mu.mu_w_ideal_assoc(in_p)
        vals_comp = [-3.77358379, -3.81520813, -4.20616932, -4.35266624]
        test_mu_1 = np.allclose(comp_mu_w_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_1)

    def test_mu_water_debye(self):
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        t_b = np.array([1e-2, 1e-2, 0.5, 1.0])

        el = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h, b_param=t_b)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[15] = test_f_b

        comp_mu_debye = el.mu_w_debye(in_p)
        vals_comp = [1.41161015e-05, 5.01848558e-03, 5.43015396e-03, 3.57547593e-03]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_comp(self):

        test_n_w = np.array([55.54, 55.0, 30.5, 30.5])
        test_n_s = np.array([0.0, 0.5, 20.0, 20.5])

        test_m = self.delta_w*test_n_s/test_n_w

        el = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h, press=0)

        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_f_b = np.array([0.0, 0.0, 0.5, 1])

        in_p = np.zeros((16,4))
        in_p[:, :] = self.in_p[:, :]
        in_p[0] = test_y
        in_p[15] = test_f_b

        comp_mu_c = el.mu_w_comp(in_p)
        vals_comp = [0.2, 0.2027027, 4.3009901, 3.88078431]
        test_mu_comp = np.allclose(comp_mu_c, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_comp)

    def test_mu_salt_ideal_assoc(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        el_mu = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([0.0, 0.0, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[0] = test_y
        in_p[1] = test_za
        in_p[2] = test_zd
        in_p[15] = test_f_b

        comp_mu_salt_1 = el_mu.mu_sf_ideal_assoc(in_p)
        vals_comp = [-36.01524939, - 27.76130061, - 23.07345586, - 16.44542837]
        test_mu_salt_1 = np.allclose(comp_mu_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_salt_1)

    def test_mu_salt_debye(self):
        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        t_b = np.array([1e-2, 1e-2, 0.5, 1.0])

        el = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h, b_param=t_b)

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[15] = np.array([0.0, 0.0, 0.25, 0.25])

        comp_mu_debye = el.mu_sf_debye(in_p)
        vals_comp = [-0.23532007, -1.66197828, -1.43308287, -1.09890055]
        test_mu_debye = np.allclose(comp_mu_debye, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_debye)

    def test_mu_salt_bjerrum_ideal_assoc(self):

        test_n_w = np.array([55.54, 55.0, 54.5, 54.5])
        test_n_s = np.array([0.01, 0.5, 1.0, 1.0])

        test_m = self.delta_w*test_n_s/test_n_w

        el_mu = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h)

        test_f_b = np.array([1e-4, 1e-4, 0.25, 0.25])
        test_y = np.array([0.6, 0.55, 0.7, 0.72])
        test_za = np.array([0.35, 0.3, 0.6, 0.55])
        test_zd = np.array([0.25, 0.2, 0.55, 0.6])

        in_p = np.zeros((16, 4))
        in_p[:, :] = self.in_p[:, :]
        in_p[0] = test_y
        in_p[1] = test_za
        in_p[2] = test_zd
        in_p[15] = test_f_b

        comp_mu_b_salt_1 = el_mu.mu_sb_ideal_assoc(in_p)
        vals_comp = [-30.16590759, -25.94904943, -14.80637131, -9.32533305]
        test_mu_b_salt_1 = np.allclose(comp_mu_b_salt_1, vals_comp, 0, 1e-6)
        self.assertTrue(test_mu_b_salt_1)

    def test_sol_pure_water(self):

        nw = 55.5
        ns = 1e-14

        ml = np.array([self.delta_w*ns/nw])

        ini_p = np.zeros(16)

        tp = np.array([300, 320, 340, 360, 380])

        for temp in tp:
            el = El.ElectrolyteSolution(ml, temp, self.param_w, self.param_salt, self.param_h)
            ini_p[0] = 0.63
            ini_p[1] = 0.4
            ini_p[2] = 0.4
            ini_p[15] = 1e-14
            sol = el.solve_eqns(ini_p, np.array([0,1,2]))
            test_cond = np.allclose(sol, el.solve_eqns_water_analytical(), 0, 1e-8)
            self.assertTrue(test_cond)

    def test_sol_water_salt_hb(self):
        def sol_salt(y, m, h):
            r = m/55.50847203605298
            t_0 = (y**2-((h[1]+2*h[2])*y-h[2])*r+(0.25*h[1]**2-h[0]*h[2])*r**2)/(1-np.sum(h)*r)
            return t_0

        nw = 55.5
        ns_mat = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        ini_p = np.zeros(16)
        ini_p[0] = 0.63
        ini_p[1] = 0.4
        ini_p[2] = 0.4
        for ns in ns_mat:
            ml = np.array([self.delta_w*ns/nw])
            el = El.ElectrolyteSolution(ml, self.temp, self.param_w, self.param_salt, self.param_h)
            ini_p[3] = 5.0
            ini_p[4] = 2.0
            ini_p[5] = 0.0
            ini_p[6] = 7.0
            ini_p[7] = 3.0
            ini_p[8] = 0.5
            ini_p[15] = 1e-14
            sol = el.solve_eqns(ini_p, np.array([0, 1, 2]))
            ini_p[:3] = sol[:]
            da = el.delta_w
            test_cond1 = np.allclose(sol[1], sol_salt(sol[0], da*ns/nw, ini_p[3:6]))
            test_cond2 = np.allclose(sol[2], sol_salt(sol[0], da*ns/nw, ini_p[6:9]))
            self.assertTrue(test_cond1)
            self.assertTrue(test_cond2)

    def test_chem_potential_infinite_dilution(self):
        # consider a negligible concentration
        ml = np.array([1e-14])

        mat_temp = np.array([300, 320, 340, 360, 380])
        ini_p = np.zeros(16)
        ini_p[0] = 0.63
        ini_p[1] = 0.4
        ini_p[2] = 0.4
        for tmp in mat_temp:
            el = El.ElectrolyteSolution(ml, tmp, self.param_w, self.param_salt, self.param_h, b_param=1.0)
            comp_analytical_water = el.mu_w0()
            comp_analytical_salt = 2*np.log(ml/el.delta_w)+el.mu_sf0()
            ini_p[15] = 1e-14
            sol = el.solve_eqns(ini_p, np.arange(15, dtype='int'))
            ini_p[:15] = sol[:]
            ini_p[15] = 1e-14
            comp_mu_water = el.mu_w(ini_p)
            test_cond = np.allclose(comp_mu_water, comp_analytical_water)
            self.assertTrue(test_cond)
            comp_mu_salt = el.mu_sf(ini_p)
            test_cond = np.allclose(comp_mu_salt, comp_analytical_salt)
            self.assertTrue(test_cond)

    def test_hydration_dilute(self):

        param_w = self.param_w
        param_w['de_w'] = 1800
        param_w['ds_w'] = 3.47

        param_salt = self.param_salt
        param_salt['de_p0'] = 1000
        param_salt['ds_p0'] = 1.0

        param_salt['de_bp0'] = 1000
        param_salt['ds_bp0'] = 1.0
        param_salt['de_bp1'] = -10000.0
        param_salt['ds_bp1'] = 0.0

        param_salt['de_m0'] = 1000
        param_salt['ds_m0'] = 1.0

        param_salt['de_bm0'] = 1000
        param_salt['ds_bm0'] = 1.0
        param_salt['de_bm1'] = -10000.0
        param_salt['ds_bm1'] = 0.0

        param_salt['de_b'] = (20+np.log(0.1))*self.temp

        m_val = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
        m_err = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
        num_eq = 16
        sol_alyt = np.zeros(16)
        ini_p = np.zeros(16)
        ini_p[0] = 0.806
        ini_p[1] = 0.65
        ini_p[2] = 0.65
        k_bjerrum = 0.2080024561114551
        for ind, ml in enumerate(m_val):
            el = El.ElectrolyteSolution(np.array([ml]), self.temp, param_w, param_salt, self.param_h, b_param=1.0)
            ini_p[3] = 2.0
            ini_p[4] = 3.0
            ini_p[5] = 0.0
            ini_p[6] = 1.16
            ini_p[7] = 4.03
            ini_p[8] = 0.0
            ini_p[9] = 1.5
            ini_p[12] = 1.64
            ini_p[15] = 1e-6
            el.define_bjerrum(k_bjerrum)
            sol = el.solve_eqns(ini_p, np.arange(num_eq, dtype='int'))
            ini_p[:num_eq] = sol[:num_eq]
            sol_alyt[:3] = el.solve_eqns_water_analytical()
            sol_alyt[3] = el.f0(el.m_p, el.f_p, el.f_p1, el.f_p2)
            sol_alyt[4] = el.f1(el.m_p, el.f_p, el.f_p1, el.f_p2)
            sol_alyt[5] = el.f2(el.m_p, el.f_p, el.f_p1, el.f_p2)
            sol_alyt[6] = el.f0(el.m_m, el.f_m, el.f_m1, el.f_m2)
            sol_alyt[7] = el.f1(el.m_m, el.f_m, el.f_m1, el.f_m2)
            sol_alyt[8] = el.f2(el.m_m, el.f_m, el.f_m1, el.f_m2)
            sol_alyt[9] = el.f0(el.m_bp, el.f_bp, el.f_bp1, el.f_bp2)
            sol_alyt[10] = el.f1(el.m_bp, el.f_bp, el.f_bp1, el.f_bp2)
            sol_alyt[11] = el.f2(el.m_bp, el.f_bp, el.f_bp1, el.f_bp2)
            sol_alyt[12] = el.f0(el.m_bm, el.f_bm, el.f_bm1, el.f_bm2)
            sol_alyt[13] = el.f1(el.m_bm, el.f_bm, el.f_bm1, el.f_bm2)
            sol_alyt[14] = el.f2(el.m_bm, el.f_bm, el.f_bm1, el.f_bm2)
            sol_alyt[15] = el.k_bjerrum0()*ml
            test_cond1 = np.allclose(sol, sol_alyt[:num_eq], 0, m_err[ind])
            self.assertTrue(test_cond1)

    def test_chem_potential_optimized_vs_non(self):

        param_w = self.param_w
        param_h = self.param_h
        param_salt = self.param_salt

        param_w['de_w'] = 1800
        param_w['se_w'] = 3.47
        param_h['m_p'] = 8.0
        param_h['m_m'] = 8.0
        param_h['mb_p'] = 4.0
        param_h['mb_m'] = 4.0

        param_salt['de_p0'] = 1000
        param_salt['ds_p0'] = 1.0
        param_salt['de_p1'] = -10000.0
        param_salt['ds_p1'] = 0.0

        param_salt['de_bp0'] = 1000
        param_salt['ds_bp0'] = 1.0
        param_salt['de_bp1'] = -10000.0
        param_salt['ds_bp1'] = 0.0

        param_salt['de_m0'] = 1000
        param_salt['ds_m0'] = 1.0
        param_salt['de_m1'] = -10000.0
        param_salt['ds_m1'] = 0.0

        param_salt['de_bm0'] = 1000
        param_salt['ds_bm0'] = 1.0
        param_salt['de_bm1'] = -10000.0
        param_salt['ds_bm1'] = 0.0

        m_val = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        num_eq = 15
        sol_alyt = np.zeros(num_eq)
        ini_p = np.zeros(16)
        ini_p[0] = 0.75
        ini_p[1] = 0.55
        ini_p[2] = 0.55
        ini_p[3] = 4.0
        ini_p[4] = 0.0
        ini_p[5] = 0.0
        ini_p[6] = 3.0
        ini_p[7] = 0.0
        ini_p[8] = 0.0
        ini_p[15] = 1e-2

        el = El.ElectrolyteSolution(m_val, self.temp, param_w, param_salt, param_h)
        sol = el.solve_eqns_multiple(ini_p, np.arange(num_eq, dtype='int'))

        mu_s_a = el.mu_sf_ideal_assoc(sol)
        mu_s_op = el.mu_sf_ideal_assoc_optimized(sol)
        mu_b_a = el.mu_sb_ideal_assoc(sol)
        mu_b_op = el.mu_sb_ideal_assoc_optimized(sol)
        test_cond1 = np.allclose(mu_s_a, mu_s_op)
        self.assertTrue(test_cond1)
        test_cond2 = np.allclose(mu_b_a, mu_b_op)
        self.assertTrue(test_cond2)

    def test_g_minus_f_equal_p(self):
        """
        tests the derivative of the free energy with respect the molality
        """
        ini_p = np.zeros(16)
        ini_p[0] = 0.4
        ini_p[1] = 0.1
        ini_p[2] = 0.1
        ini_p[3] = 5.0
        ini_p[4] = 1.0
        ini_p[5] = 0.0
        ini_p[6] = 5.0
        ini_p[7] = 1.0
        ini_p[8] = 0.5
        ini_p[9] = 3.0
        ini_p[10] = 1.0
        ini_p[11] = 0.0
        ini_p[12] = 3.0
        ini_p[13] = 1.0
        ini_p[14] = 0.0
        ini_p[15] = 1e-1

        test_m = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1.4, 2.0])

        el_m_a = El.ElectrolyteSolution(test_m, self.temp, self.param_w, self.param_salt, self.param_h, b_param=1)
        mu_w = el_m_a.mu_w_ideal_assoc(ini_p)
        mu_sf = el_m_a.mu_sf_ideal_assoc(ini_p)
        mu_sb = el_m_a.mu_sb_ideal_assoc(ini_p)

        g_en = el_m_a.n_w*mu_w+el_m_a.n_s*((1-ini_p[15])*mu_sf+ini_p[15]*mu_sb)
        f_en = el_m_a.f_ideal()+el_m_a.f_assoc(ini_p)

        p_id_1 = el_m_a.n_w+2*(1-ini_p[15])*el_m_a.n_s+ini_p[15]*el_m_a.n_s
        p_id_2 = -2*ini_p[0]*el_m_a.n_w
        p_id_3 = -((1-ini_p[15])*np.sum(ini_p[3:9])+ini_p[15]*np.sum(ini_p[9:15]))*el_m_a.n_s
        p_id = p_id_1+p_id_2+p_id_3

        dp = g_en-f_en
        test_cond = np.allclose(dp, p_id, 0, 1e-14)
        self.assertTrue(test_cond)

    def test_chem_obtained_from_f_numerically(self):
        """
        compare the numerical evaluation of the chemical potential with the actual formula
        """

        temp = 298.15
        v_w = 30.0
        v_s = 27.0

        de_w = 1800
        ds_w = 3.47
        de = 1000.0
        d2e = -10000.0
        mr_p, mr_m, mr_bp, mr_bm = [8.0, 8.0, 8.0, 8.0]

        param_w = {'v_w': v_w, 'de_w': de_w, 'ds_w': ds_w, 'de_2d': 0.0, 'ds_2d': 0.0, 'de_2a': 0.0, 'ds_2a': 0.0}
        dict_vol = {'v_s': v_s, 'v_b': v_s}
        dict_p = {'de_p0': de, 'ds_p0': 0.0, 'de_p1': d2e, 'ds_p1': 0.0, 'de_p2': d2e, 'ds_p2': 0.0}
        dict_bp = {'de_bp0': de, 'ds_bp0': 0.0, 'de_bp1': d2e, 'ds_bp1': 0.0, 'de_bp2': d2e, 'ds_bp2': 0.0}
        dict_m = {'de_m0': de, 'ds_m0': 0.0, 'de_m1': d2e, 'ds_m1': 0.0, 'de_m2': d2e, 'ds_m2': 0.0}
        dict_bm = {'de_bm0': de, 'ds_bm0': 0.0, 'de_bm1': d2e, 'ds_bm1': 0.0, 'de_bm2': d2e, 'ds_bm2': 0.0}
        dict_b = {'de_b': np.log(0.7), 'ds_b': 0.0}
        param_salt = {**dict_vol, **dict_p, **dict_bp, **dict_m, **dict_bm, **dict_b}
        dict_max = {'m_p': mr_p, 'm_m': mr_m, 'mb_p': mr_bp, 'mb_m': mr_bm}
        param_h = {**dict_max}

        ini_val = np.array([10.0, 10.0, 922.2222222222222, 671.1111111111111, -10000.0, -10000.0, 1e-10])
        ini_p0 = np.array([0.75, 0.55, 0.55, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1e-14])

        b_par = 1.0
        param_h['m_p'] = ini_val[0]
        param_h['m_m'] = ini_val[1]
        param_h['mb_p'] = ini_val[0] - 1
        param_h['mb_m'] = ini_val[1] - 1
        param_salt['de_p0'] = ini_val[2]
        param_salt['de_m0'] = ini_val[3]
        param_salt['de_bp0'] = ini_val[2]
        param_salt['de_bm0'] = ini_val[3]
        param_salt['de_p1'] = ini_val[4]
        param_salt['de_m1'] = ini_val[5]
        param_salt['de_bp1'] = ini_val[4]
        param_salt['de_bm1'] = ini_val[5]

        num_pnts = 20
        m_val = np.linspace(1e-2, 1.0, num_pnts)
        dm = 1e-9

        el_p = El.ElectrolyteSolution(m_val+dm, temp, param_w, param_salt, param_h, b_param=b_par)
        el_p.define_bjerrum(ini_val[6])
        el_m = El.ElectrolyteSolution(m_val-dm, temp, param_w, param_salt, param_h, b_param=b_par)
        el_m.define_bjerrum(ini_val[6])
        delta_w = el_p.delta_w

        # redefine the variables so that the chemical potential is computed
        c_fac=0.97
        # salt chemical potential
        delta_w = el_p.delta_w
        el_p.n_w = c_fac*np.ones_like(el_p.ml)
        el_p.n_s = (el_p.ml/delta_w)*el_p.n_w
        el_m.n_w = c_fac*np.ones_like(el_m.ml)
        el_m.n_s = (el_m.ml/delta_w)*el_m.n_w

        sol = el_p.solve_eqns_multiple(ini_p0, np.arange(16, dtype='int'))
        sol[15, :]=0.0

        # free salt chemical potential
        mu_c_assoc = delta_w*(el_p.f_assoc(sol)-el_m.f_assoc(sol))/(2*dm*c_fac)
        mu_c_ideal = delta_w*(el_p.f_ideal()-el_m.f_ideal())/(2*dm*c_fac)
        mu_c_elec = delta_w*(el_p.f_debye(sol)-el_m.f_debye(sol))/(2*dm*c_fac)

        mu_f_idassoc = el_p.mu_sf_ideal_assoc(sol)
        mu_f_elec = el_p.mu_sf_debye(sol)

        test_electro = np.allclose(mu_c_elec, mu_f_elec, 0, 1e-5)
        self.assertTrue(test_electro)

        test_idassoc = np.allclose(mu_c_ideal + mu_c_assoc, mu_f_idassoc, 0, 8e-5)
        self.assertTrue(test_idassoc)

        # bjerrum chemical potential
        solb = np.zeros(16)
        solb[0] = 0.4
        solb[1] = 0.2
        solb[2] = 0.2
        solb[9] = 4.0
        solb[10] = 2.0
        solb[11] = 1.0
        solb[12] = 5.0
        solb[13] = 3.0
        solb[14] = 0.5
        solb[15] = 1.0

        mu_c_assoc = delta_w*(el_p.f_assoc(solb)-el_m.f_assoc(solb))/(2*dm*c_fac)
        mu_c_ideal = delta_w*(el_p.f_ideal()-el_m.f_ideal())/(2*dm*c_fac)
        mu_f_idassoc = el_p.mu_sb_ideal_assoc(solb)

        test_idassoc = np.allclose(mu_c_ideal + mu_c_assoc, mu_f_idassoc, 0, 5e-5)
        self.assertTrue(test_idassoc)

        # water chemical potential
        c_fac = 0.1
        el_p.n_s = c_fac*np.ones_like(el_p.ml)
        el_p.n_w = delta_w*el_p.n_s/el_p.ml

        el_m.n_s = c_fac*np.ones_like(el_m.ml)
        el_m.n_w = delta_w*el_m.n_s/el_m.ml

        mu_c_assoc = -el_p.ml**2*(el_p.f_assoc(sol) - el_m.f_assoc(sol)) / (2*delta_w*el_p.n_s*dm)
        mu_c_ideal = -el_p.ml**2*(el_p.f_ideal() - el_m.f_ideal()) / (2*delta_w*el_p.n_s*dm)
        mu_c_elec = -el_p.ml**2*(el_p.f_debye(sol) - el_m.f_debye(sol)) / (2*delta_w*el_p.n_s*dm)

        mu_f_idassoc = el_p.mu_w_ideal_assoc(sol)
        mu_f_elec = el_p.mu_w_debye(sol)
        test_electro = np.allclose(mu_c_elec, mu_f_elec, 0, 1e-5)
        self.assertTrue(test_electro)

        test_idassoc = np.allclose(mu_c_ideal+mu_c_assoc, mu_f_idassoc, 0, 5e-6)
        self.assertTrue(test_idassoc)

    def test_gibbs_duhem(self):
        """
        Tests the Gibbs Duhem Relation
        """

        temp = 298.15
        v_w = 30.0
        v_s = 27.0

        de_w = 1800
        ds_w = 3.47
        de = 1000.0
        d2e = -10000.0
        mr_p, mr_m, mr_bp, mr_bm = [8.0, 8.0, 8.0, 8.0]

        param_w = {'v_w': v_w, 'de_w': de_w, 'ds_w': ds_w, 'de_2d': 0.0, 'ds_2d': 0.0, 'de_2a': 0.0, 'ds_2a': 0.0}
        dict_vol = {'v_s': v_s, 'v_b': v_s}
        dict_p = {'de_p0': de, 'ds_p0': 0.0, 'de_p1': d2e, 'ds_p1': 0.0, 'de_p2': d2e, 'ds_p2': 0.0}
        dict_bp = {'de_bp0': de, 'ds_bp0': 0.0, 'de_bp1': d2e, 'ds_bp1': 0.0, 'de_bp2': d2e, 'ds_bp2': 0.0}
        dict_m = {'de_m0': de, 'ds_m0': 0.0, 'de_m1': d2e, 'ds_m1': 0.0, 'de_m2': d2e, 'ds_m2': 0.0}
        dict_bm = {'de_bm0': de, 'ds_bm0': 0.0, 'de_bm1': d2e, 'ds_bm1': 0.0, 'de_bm2': d2e, 'ds_bm2': 0.0}
        dict_b = {'de_b': np.log(0.7), 'ds_b': 0.0}
        param_salt = {**dict_vol, **dict_p, **dict_bp, **dict_m, **dict_bm, **dict_b}
        dict_max = {'m_p': mr_p, 'm_m': mr_m, 'mb_p': mr_bp, 'mb_m': mr_bm}
        param_h = {**dict_max}

        ini_val = np.array([10.0, 10.0, 922.2222222222222, 671.1111111111111, -10000.0, -10000.0, 0.05])
        ini_p0 = np.array([0.75, 0.55, 0.55, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1e-14])

        b_par = 1.0
        param_h['m_p'] = ini_val[0]
        param_h['m_m'] = ini_val[1]
        param_h['mb_p'] = ini_val[0] - 1
        param_h['mb_m'] = ini_val[1] - 1
        param_salt['de_p0'] = ini_val[2]
        param_salt['de_m0'] = ini_val[3]
        param_salt['de_bp0'] = ini_val[2]
        param_salt['de_bm0'] = ini_val[3]
        param_salt['de_p1'] = ini_val[4]
        param_salt['de_m1'] = ini_val[5]
        param_salt['de_bp1'] = ini_val[4]
        param_salt['de_bm1'] = ini_val[5]

        num_pnts=20
        m_val = np.linspace(1e-2, 2.0, num_pnts)
        dm = 1e-7
        el_p = El.ElectrolyteSolution(m_val+dm, temp, param_w, param_salt, param_h, b_param=b_par)
        el_p.define_bjerrum(ini_val[6])
        el_m = El.ElectrolyteSolution(m_val-dm, temp, param_w, param_salt, param_h, b_param=b_par)
        el_m.define_bjerrum(ini_val[6])

        delta_w = el_p.delta_w

        sol_p = el_p.solve_eqns_multiple(ini_p0, np.arange(16, dtype='int'))
        sol_m = el_m.solve_eqns_multiple(ini_p0, np.arange(16, dtype='int'))

        mu_s_p = (1-sol_p[15])*el_p.mu_sf(sol_p)+sol_p[15]*el_p.mu_sb(sol_p)
        mu_s_m = (1-sol_m[15])*el_m.mu_sf(sol_m)+sol_m[15]*el_m.mu_sb(sol_m)

        der_mu_water = (el_p.mu_w(sol_p)-el_m.mu_w(sol_m))/(2*dm)
        der_mu_salt_f = (el_p.mu_sf(sol_p)-el_m.mu_sf(sol_m))/(2*dm)
        der_mu_salt_b = (el_p.mu_sb(sol_p)-el_m.mu_sb(sol_m))/(2*dm)

        der_f = (el_p.f_total(sol_p)-el_m.f_total(sol_m))/(2*dm)
        der_f_2 = (el_p.f_total(sol_p)-el_m.f_total(sol_p))/(2*dm)
        der_f_ideal = (el_p.f_ideal()-el_m.f_ideal())/(2*dm)
        der_f_assoc = (el_p.f_assoc(sol_p)-el_m.f_assoc(sol_m))/(2*dm)
        der_f_assoc_2 = (el_p.f_assoc(sol_p) - el_m.f_assoc(sol_p)) / (2 * dm)
        der_w = (el_p.n_w-el_m.n_w)/(2*dm)
        der_s  = (el_p.n_s-el_m.n_s)/(2*dm)

        print('assoc')
        print(der_f_ideal)
        print(der_f_assoc_2)
        print(der_f_assoc)

        print('check equations')
        for ind in range(16):
            print(ind, np.max(np.abs(el_p.eqns(sol_p[:, ind], ind))))
            print(np.max(np.abs(el_m.eqns(sol_m[:, ind], ind))))

        print(der_f_2)
        print(sol_p.shape)
        for ind in range(16):
            sol_new = sol_m[:, :]
            #sol_new[ind, :] = sol_p[ind, :]
            der_f_ind = (el_p.f_total(sol_p)-el_m.f_total(sol_new))/(2*dm)
            #print(der_f_ind-der_f_2)
            print(ind, np.max(np.abs(der_f_ind-der_f_2)))

        print('simple derivatives')
        print(der_w)
        print(der_s)

        der_w_cons = delta_w*der_mu_water/m_val
        der_mu_s_p = (1-sol_p[15])*der_mu_salt_f+sol_p[15]*der_mu_salt_b
        der_mu_s_m = (1-sol_m[15])*der_mu_salt_f+sol_m[15]*der_mu_salt_b

        dp_1 = -der_f+el_p.mu_w(sol_p)*der_w+mu_s_p*der_s
        dm_1 = -der_f+el_m.mu_w(sol_m)*der_w+mu_s_m*der_s

        mu_v_sf = el_p.mu_sf_ideal_assoc(sol_p)+el_p.mu_sf_debye(sol_p)
        mu_v_sb = el_p.mu_sb_ideal_assoc(sol_p)
        mu_v_w = el_p.mu_w_ideal_assoc(sol_p)+el_p.mu_w_debye(sol_p)
        # first identity
        test_first = np.allclose(dp_1, 0.0, 0.0, 3e-3)
        self.assertTrue(test_first)

        delta_comp = (der_w+der_s*el_p.u_s/el_p.u_w)*el_p.mu_w_comp(sol_p)
        print('f derivative')
        print(der_f_2)
        print('delta_f')
        delta_f = der_f_2-mu_v_w*der_w-((1-sol_p[15])*mu_v_sf+sol_p[15]*mu_v_sb)*der_s
        print(der_f)
        print(delta_f)
        print('mu')
        print(el_p.mu_w_comp(sol_p)*(der_w+der_s*el_p.u_s/el_p.u_w))
        print('mu')
        print(mu_v_sb-mu_v_sf)
        dp_2 = der_mu_water*el_p.n_w+ der_mu_s_p*el_p.n_s
        dm_2 = der_mu_water*el_m.n_w+ der_mu_s_m*el_m.n_s
        print('f_com')
        print(el_p.f_comp(sol_p))
        print(el_m.f_comp(sol_m))
        # second identity
        test_second = np.allclose(dp_2, 0.0, 0.0, 3e-3)
        self.assertTrue(test_second)

        # pressure
        test_pressure = np.allclose((dp_2+dm_2+dp_1+dm_1)*0.5, 0.0, 0.0, 1e-7)
        self.assertTrue(test_pressure)



if __name__ == '__main__':
    unittest.main()
