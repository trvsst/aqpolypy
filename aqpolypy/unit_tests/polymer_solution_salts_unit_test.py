"""
:module: polymer_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: CHI YUANCHEN <ychi@iastate.edu>, Jun 020
.. history:
"""
import unittest
import numpy as np

import aqpolypy.free_energy_polymer.PolymerSolutionSalts as Pss


import aqpolypy.salts_theory.Bjerrum as bj
import aqpolypy.water.WaterMilleroAW as aw


class TestPolymerwithSalts(unittest.TestCase):

    def test_free_c(self):
        """ checks free energy when there is no salt
        """

        num_pnts = 10
        f_comp = np.array([-3.27242457671423, -3.04309724043625, -2.81415966950124, -2.58499055405458, -2.35477419179418, -2.12242993842406,
                           -1.88647609090591, -1.64476928859107, -1.39395014167604, -1.12803135003374])
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([1e-12, 1, 1, -100/3, -100/3])
        v_w = 1000
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-7))

    def test_free_p(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10

        f_comp = np.array([-2.62241014447868, -2.39436157978325, -2.16681661968465, -1.93917561583082, -1.71065102238454, -1.48019941152741,
                           -1.24638883097259, -1.00714166588399, -0.759175894742952, -0.496523119999612]) - 1.1894435910007868e-07

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.002, 1, 1, -100/3, -100/3])
        v_w = 1000
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([3.30667242959371, 3.52579321001907, 3.74402184043587, 3.96203065390970, 4.18079591865398, 4.40181270186095,
                           4.62761376840203, 4.86322215062866, 5.12157703934959, 5.46507946701045]) - 1.1894435910007868e-07

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.02, 1, 1, -100/3, -100/3])
        v_w = 1000
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-5))

    def test_potential_w(self):
        """
            checks chemical potential of water as a function of math:`\\phi_p`
        """

        num_pnts = 10

        p_comp = np.array([-3.56439469989, -3.56309084629, -3.56178725172, -3.56048391744, -3.55918084473, \
                           -3.55787803486, -3.55657548914, -3.55527320883, -3.55397119523, -3.55266944962])

        v_p = np.array([1e-12, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1000;
        df_w = 10/3;
        x_ini = 0.1;
        p_ini = 0.2;
        n_k = 100;
        chi_p = 0.5;
        chi_e = 0.5;
        param_s = np.array([7, 7, 1, 1]);

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac=np.array([0, 0])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([0, 0])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)/100
        potential = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential[ind] = polymer_sol.chem_potential_w()

        self.assertTrue(np.allclose(potential, p_comp, rtol=0.0, atol=1e-3))

    def test_potential_w_s(self):
        """
            checks chemical potential of water as a function of math:`c_s`
        """

        num_pnts = 10

        p_comp = np.array([-3.56049354763, -3.55960679586, -3.55879411401, -3.5580569097, -3.5573966216,\
                           -3.55681472029, -3.55631270922, -3.55589212563, -3.55555454156, -3.55530156489])

        v_p = np.array([0.004, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1000;
        df_w = 10/3;
        x_ini = 0.1;
        p_ini = 0.2;
        n_k = 100;
        chi_p = 0.5;
        chi_e = 0.5;
        param_s = np.array([7, 7, 1, 1]);

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac=np.array([0, 0])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([0, 0])

        c_val = np.linspace(1e-1, 0.8, num_pnts)/100
        potential = np.zeros_like(c_val)

        for ind, c_s in enumerate(c_val):
            polymer_sol = Pss.PolymerSolutionSalts(v_p, np.array([c_s, 1, 1, -100/3, -100/3]), v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential[ind] = polymer_sol.chem_potential_w()

        self.assertTrue(np.allclose(potential, p_comp, rtol=0.0, atol=2*1e-3))        
   

if __name__ == '__main__':
    unittest.main()
