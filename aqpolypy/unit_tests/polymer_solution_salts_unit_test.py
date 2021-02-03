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
        temp = 298
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
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_p(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10

        #f_comp = np.array([-2.62241014447868, -2.39436157978325, -2.16681661968465, -1.93917561583082, -1.71065102238454, -1.48019941152741,
        #                   -1.24638883097259, -1.00714166588399, -0.759175894742952, -0.496523119999612]) - 1.1894435910007868e-07


        f_comp = np.array([-3.25435775411, -3.02655415914, -2.79914382161, -2.57150579465, -2.34282482276, 
                           -2.11202082484, -1.87761282905, -1.6374584606, -1.38819972495, -1.12385141787]) - 1.2e-07

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.002 , 1, 1, -100/3, -100/3])
        temp = 298
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
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-8))

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([14.2490865827, 12.9997832414, 11.7468389769, 10.4905389126, 9.23128576646, 
                           7.96964119543, 6.70641631518, 5.44285800741, 4.18107853461, 2.92527560925]) - 0.00288182 
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2 , 1, 1, -100/3, -100/3])
        temp = 298
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
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-8))

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



        ANS = np.array([[-3.4038360991393108770353657932528,  -626.71627660122857246538519149226,  274.16364121466509794211585671064],
                        [-3.2809280725663454313556673114238,  -582.83991369738981559722734715479,  274.95893825671441935976080372939],
                        [-3.1646417928644517617692809297036,  -537.89729872475573667356396612149,  275.8190631044749715553021651715],
                        [-3.057749020708892530371995899241,  -491.80311984672434119647554950916,  276.75823568053715358072697227954],
                        [-2.9641356490979617047316387046241,  -444.54525665968310005338182122614,  277.7964426414147302915602644191],
                        [-2.8895069610317752620753592451841,  -396.12794477287499235462764346494,  278.96302912832365576006887764038],
                        [-2.8427607994324602168974454319272,  -346.54804655807527472011209113134,  280.30360087186457980490406072249],
                        [-2.8389666863811733754250840600086,  -295.76305257915517910732861039946,  281.89473148352634548213790921523],
                        [-2.9068936548941135764858798018473,  -243.61036790516253041846694848971,  283.880207174370101957938672399],
                        [-3.1130697936646442539488505163447,  -189.52960388084491790208763184467,  286.582761105125007658478430983]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
         

        v_p = np.array([1e-12, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        temp = 298;
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
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
   
        dw = np.array([2.27342744e-06, 2.06671231e-06, 1.85999949e-06, 1.65328900e-06,
                       1.44658082e-06, 1.23987496e-06, 1.03317141e-06, 8.26470183e-07,
                       6.19771272e-07, 4.13074677e-07])
        dp = np.array([0.00068203, 0.00062001, 0.000558  , 0.00049599, 0.00043397,
                       0.00037196, 0.00030995, 0.00024794, 0.00018593, 0.00012392])
        dpm = np.array([-0.03688068, -0.03688068, -0.03688068, -0.03688068, -0.03688068,
                        -0.03688068, -0.03688068, -0.03688068, -0.03688068, -0.03688068])
                                
        self.assertTrue(np.allclose(potential_w, w_comp + dw , rtol=0.0, atol=1e-14))
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-8))        
        self.assertTrue(np.allclose(potential_pm, pm_comp + dpm, rtol=0.0, atol=1e-8))

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


        ANS = np.array([[-3.5587750593984337520235376739475, -668.32611301628401433150283850182, 277.53966495830262847105407431059],
                        [-3.5574902217880475332502376861754, -657.83677674585519223882101869094, 278.41465306128719633255164112029],
                        [-3.5571262925232590222223566511772, -647.36889610211896094182915106785, 279.08154101502817542174161857815],
                        [-3.5577096306307680202143592373965, -636.92132053912444224208488918748, 279.65575734793225701202423749692],
                        [-3.5592681862855609164846514003155, -626.49287456717725341981406472769, 280.17870396578415163879594018681],
                        [-3.5618316266781097137179672162954, -616.08235611124930004856015131054, 280.67010773868742092712613711569],
                        [-3.5654314751493801663407776092463, -605.68853476408516606221965616896, 281.14098654521980621967536251304],
                        [-3.5701012653476875980665047882745, -595.31014992528852283748541651676, 281.59817503894961956967257776796],
                        [-3.5758767124433255079436408396809, -584.94590881682858149459569574447, 282.04625016423022533518238930128],
                        [-3.5827959037725266886278453759562, -574.59448436447739824717295675782, 282.48846622063717743539301707756]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
     
        dw = np.array([0.0008397 , 0.00194376, 0.00328821, 0.00482154, 0.00651263,
                       0.00834013, 0.01028824, 0.01234471, 0.01449968, 0.01674504])
        dp = np.array([0.25190896, 0.58312785, 0.98646274, 1.4464614 , 1.95378938,
                       2.50203844, 3.08647152, 3.70341187, 4.34990481, 5.02351314])
        dpm = np.array([-0.24515244, -0.31932363, -0.37589445, -0.42267796, -0.46302142,
                        -0.49872392, -0.53088351, -0.5602281 , -0.58726925, -0.61238243])

        v_p = np.array([0.004, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        temp = 298;
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
            polymer_sol = Pss.PolymerSolutionSalts(v_p, np.array([c_s, 1, 1, -100/3, -100/3]), temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()

        self.assertTrue(np.allclose(potential_w, w_comp + dw, rtol=0.0, atol=1e-8))
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-8))        
        self.assertTrue(np.allclose(potential_pm, pm_comp + dpm, rtol=0.0, atol=1e-8))
        
    def test_potential_df(self):
        """
            checks if chemical potentials add up give freee nergy
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
        temp = 298;
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
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)
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


            D_w = 55.509

            # volume fractions       

            V_a = conc / u_a / D_w 
            V_b = conc / u_b / D_w 
            V_w = 1 
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
