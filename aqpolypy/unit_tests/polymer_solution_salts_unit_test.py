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

import aqpolypy.units.units as un


class TestPolymerwithSalts(unittest.TestCase):

    def test_free_c(self):
        """ checks free energy when there is no salt
        """

        num_pnts = 10
        #f_comp = np.array([-3.27242457671423, -3.04309724043625, -2.81415966950124, -2.58499055405458, -2.35477419179418, -2.12242993842406,
                           #-1.88647609090591, -1.64476928859107, -1.39395014167604, -1.12803135003374])
        f_comp = np.array([-3.27242457671, -3.04309724043, -2.81415966949, -2.58499055405, -2.35477419179, 
                           -2.12242993842, -1.8864760909, -1.64476928859, -1.39395014167, -1.12803135003])                   
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([1e-12, 1, 1, -100/3, -100/3])
        v_w = 1000
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_p(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10

        #f_comp = np.array([-2.62241014447868, -2.39436157978325, -2.16681661968465, -1.93917561583082, -1.71065102238454, -1.48019941152741,
        #                   -1.24638883097259, -1.00714166588399, -0.759175894742952, -0.496523119999612]) - 1.1894435910007868e-07


        f_comp = np.array([-3.2541077688, -3.02632566569, -2.7989368304, -2.57132031715, -2.34266087176,
                           -2.11187841481, -1.87749197665, -1.6373591854, -1.38812205068, -1.1237953745])+\
                 np.array([-1.96167772e-04, -1.79225281e-04, -1.62282790e-04, -1.45340298e-04,
                           -1.28397807e-04, -1.11455316e-04, -9.45128245e-05, -7.75703332e-05,
                           -6.06278419e-05, -4.36853506e-05])
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.002, 1, 1, -100/3, -100/3])
        v_w = 1
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-9))

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([14.4816154981, 13.2123176787, 11.9393699444, 10.6630564967, 9.38377892507,
                           8.10209747504, 6.81882145408, 5.53519536283, 4.25332825777, 2.97741368404]) +\
                 np.array([-0.18589876, -0.1700971 , -0.15429545, -0.13849379, -0.12269214,
                           -0.10689048, -0.09108883, -0.07528717, -0.05948552, -0.04368386])

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2, 1, 1, -100/3, -100/3])
        v_w = 1
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-4))

    def test_potential_w(self):
        """
            checks chemical potential as a function of math:`\\phi_p`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_pm = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)
        pm_comp = np.zeros_like(phi_val)



        ANS = np.array([[-3.4038361493523114994119889431534,  -626.7155484051118973787275479026,  274.16663381125029018414718251506 ],
                        [-3.2809283410530498928023811211485,  -582.83924791203465949618323094228,  274.9619306810961522416492108789 ],
                        [-3.1646422811651264014550823111538,  -537.89669513401170516941895982654,  275.82205536528643319976739078925 ],
                        [-3.0577497311063017228520075074805,  -491.802578204651825201365600293,  276.76122778945711327275303048623 ],
                        [-2.9641365850602121705939170914768,  -444.54477668026334642725760915027,  277.79943461487184314107068416888 ],
                        [-2.8895081280273319434214428225172,  -396.12752611320760434687748841043,  278.96602099060105486860727808107 ],
                        [-2.8427622065624429430674355301356,  -346.54768878839477013603653787754,  280.30659266126266680047018908795 ],
                        [-2.8389683500451172024807780513989,  -295.7627551224155234912971401684,  281.89772326609918356998818467218 ],
                        [-2.906895608498743290362686941819,  -243.61012989197610099229018842948,  283.88319908008204223159864909576 ],
                        [-3.1130721210598222121156142261356,  -189.52942368998682506203712350157,  286.58575345176894180049265159482 ]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
         

        v_p = np.array([1e-12, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1;
        df_w = 10/3;
        x_ini = 0.1;
        p_ini = 0.2;
        n_k = 100;
        chi_p = 0.5;
        chi_e = 0.5;
        param_s = np.array([7, 7, 1, 1, 8, 8]);

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac=np.array([1, 1])



        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
   
        self.assertTrue(np.allclose(potential_w, w_comp + np.array([2.28352520e-06, 2.07589204e-06, 1.86826121e-06, 1.66063271e-06,
                        1.45300655e-06, 1.24538272e-06, 1.03776123e-06, 8.30142072e-07,
                        6.22525248e-07, 4.14910757e-07]), rtol=0.0, atol=1e-14))
        self.assertTrue(np.allclose(potential_p, p_comp + np.array([0.00068506, 0.00062277, 0.00056048, 0.00049819, 0.0004359 ,
                        0.00037361, 0.00031133, 0.00024904, 0.00018676, 0.00012447]), rtol=0.0, atol=1e-8))        
        self.assertTrue(np.allclose(potential_pm, pm_comp + np.array([-0.03693459, -0.03693459, -0.03693459, -0.03693459, -0.03693459,
                        -0.03693459, -0.03693459, -0.03693459, -0.03693459, -0.03693459]), rtol=0.0, atol=1e-6))

    def test_potential_w_s(self):
        """
            checks chemical potential of water as a function of math:`c_s`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_pm = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)
        pm_comp = np.zeros_like(phi_val)


        ANS = np.array([[-3.5587684226730172791248324459801,  -668.28595072849411154458643125054,  277.54377333157565662519195015534],
                        [-3.5574846085003878321624190550487,  -657.76552707520624431458688974317,  278.41967560059221525187167500803],
                        [-3.5571273658891313902501683796853,  -647.26667909346270529669098436898,  279.08750484118182426385157413051],
                        [-3.5577233027119007829951538579571,  -636.78824573536715464294744037943,  279.66269120423019520216278976932],
                        [-3.5593006382050672830588342043789,  -626.3290406952882641150592846202,  280.18663834545652254609526821127],
                        [-3.5618893308638730354370206809822,  -615.88785074253941445587834380149,  280.67907502566598768045658118808],
                        [-3.5655212198654906236776167272051,  -605.46343394675143834149366739439,  281.15102116995883185096571921235],
                        [-3.5702301818147286767532629092603,  -595.05451778702760073924693529079,  281.60931365131962718362235165159],
                        [-3.5760523048965821788771832223652,  -584.65979713510453437408376231588,  282.05853182583295844787496520212],
                        [-3.5830260828646919304355661362038,  -574.27793210178842057966885903397,  282.50193261928290482110082137979]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
     
        dw = np.array([0.00084394, 0.00195457, 0.00330816, 0.00485318, 0.00655857,
                       0.00840302, 0.01037078, 0.01244965, 0.01462981, 0.01690318])
        dp = np.array([0.25318304, 0.58637237, 0.99244665, 1.4559543 , 1.96757234,
                       2.52090673, 3.11123483, 3.73489374, 4.38894166, 5.07095333])                  
        dpm = np.array([-0.24568162, -0.32019591, -0.37713575, -0.42431342, -0.46507381,
                        -0.50121373, -0.53382924, -0.56364657, -0.59117584, -0.61679129])

        v_p = np.array([0.004, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1;
        df_w = 10/3;
        x_ini = 0.1;
        p_ini = 0.2;
        n_k = 100;
        chi_p = 0.5;
        chi_e = 0.5;
        param_s = np.array([7, 7, 1, 1, 8 ,8]);

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac=np.array([1, 1])


        c_val = np.linspace(0.1, 0.8, num_pnts)

        for ind, c_s in enumerate(c_val):
            polymer_sol = Pss.PolymerSolutionSalts(v_p, np.array([c_s, 1, 1, -100/3, -100/3]), v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()

        self.assertTrue(np.allclose(potential_w, w_comp + dw, rtol=0.0, atol=1e-8))
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-8))        
        self.assertTrue(np.allclose(potential_pm, pm_comp + dpm, rtol=0.0, atol=1e-8))
        
    def test_potential_df(self):
        """
            checks eqn(132)
        """
        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_pm = np.zeros_like(phi_val)
        f_com = np.zeros_like(phi_val)
        F_com = np.zeros_like(phi_val)
        
        v_p = np.array([0.004, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1000;
        df_w = 10/3;
        x_ini = 0.1;
        p_ini = 0.2;
        n_k = 100;
        chi_p = 0.5;
        chi_e = 0.5;
        param_s = np.array([7, 7, 1, 1, 8, 8]);

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac=np.array([1, 1])

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
            f_com[ind] = polymer_sol.free()
   
            # concentration in mols/litre
            conc = v_s[0]

            # molecular volumes
            u_p = v_p[1]
            u_a = v_s[1]
            u_b = v_s[2]

            v_a = 1.8068689246447816e-05 / u_a  # m^3/mol
            v_b = 1.8068689246447816e-05 / u_b

            # volume fractions

            V_a = conc * v_a * v_w # m^3
            V_b = conc * v_b * v_w # m^3
            V_w = v_w * 1e-3 # m^3
            V_all = (V_a + V_b + V_w) / (1 - phi_p)

            phi_a = V_a / V_all
            phi_b = V_b / V_all
            phi_w = 1-phi_a-phi_b-phi_p#V_w / V_all
            phi_1 = phi_a + phi_b

            
            F_com[ind] = (u_p / n_k * phi_p * potential_p[ind]
                         + u_a * phi_a * potential_pm[ind]
                         + u_a * phi_b * potential_pm[ind]
                         + phi_w * potential_w[ind])

        self.assertTrue(np.allclose(F_com, f_com, rtol=0.0, atol=1e-15))

        
if __name__ == '__main__':
    unittest.main()
