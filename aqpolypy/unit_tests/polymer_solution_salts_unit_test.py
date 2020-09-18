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
        param_s = np.array([7, 7, 1, 1, 8, 8])

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

        #f_comp = np.array([-2.62241014447868, -2.39436157978325, -2.16681661968465, -1.93917561583082, -1.71065102238454, -1.48019941152741,
        #                   -1.24638883097259, -1.00714166588399, -0.759175894742952, -0.496523119999612]) - 1.1894435910007868e-07
        f_comp = np.array([-2.59230329872,
                           -2.36119806282,
                           -2.13029151457,
                           -1.89891636198,
                           -1.66619206524,
                           -1.43094235247,
                           -1.19153595328,
                           -0.945576056776,
                           -0.689220713748,
                           -0.415374558496]) - 1.1894467355499927e-07

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.002, 1, 1, -100/3, -100/3])
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
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-6))

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([3.68859304612,
                 3.93907337292,
                 4.19187942152,
                 4.44842022342,
                 4.71069483338,
                 4.98167766152,
                 5.26615740695,
                 5.57285091264,
                 5.92138984185,
                 6.38836556906])-1.1894145780808949e-07-2.9012919891820205e-12

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.02, 1, 1, -100/3, -100/3])
        v_w = 1000
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_e = 0.5
        param_s = np.array([7, 7, 0, 0, 8, 8])

        wa = aw.WaterPropertiesFineMillero(tk=300, pa=1)
        b_o = bj.Bjerrum(wa)
        b_fac = np.array([1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, v_w, df_w, x_ini, p_ini, n_k, chi_p, chi_e, param_s, b_o, b_fac)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-1))

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


        ANS = np.array([[-3.4048244660368856861804194580134,  -617.85383471586658157836957627751,  278.05259473915852096170819862664],
                        [-3.2849695988418769117831233494975,  -573.94766544169590982704676207504,  278.96289864233787725300572901688],
                        [-3.172424428192501822667756505858,  -528.97012468628369342982242784501,  279.95234645194552722952119339084],
                        [-3.0702189737323660265059036000856,  -482.8335538757566854847769199921,  281.03908911982613223756732878655],
                        [-2.9826455478062045897501460001155,  -435.52202231639749083202950430405,  282.24886339392556218463870010282],
                        [-2.916088115816548221865926737717,  -387.03320844694543467062741726181,  283.61987326426544621382763722295],
                        [-2.8806693610749817556536747642504,  -337.35179250288148555120271154806,  285.21233371561713637598600712408],
                        [-2.893917300609197515253150773118,  -286.41011706452727542809855165729,  287.1292660650605725811135293668],
                        [-2.9903664750497109936890408055099,  -233.9849649741131007857353196755,  289.56933319851635075248275086324],
                        [-3.2539724479408136392777350920369,  -179.3274344967701203999721482063,  292.9973758415054955264733571596]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
         

        v_p = np.array([1e-12, 1/3, 10/3]);
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

        self.assertTrue(np.allclose(potential_w, w_comp + 8.90656588e-05, rtol=0.0, atol=1e-13))
        self.assertTrue(np.allclose(potential_p, p_comp + 0.02671969763517588, rtol=0.0, atol=1e-12))        
        self.assertTrue(np.allclose(potential_pm, pm_comp - 0.03693459, rtol=0.0, atol=1e-10))

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

#        ANS = np.array([[-3.5578006665281543679526209122965,  -636.06247287131324819053901364896,  282.7145316966482658793680116105],
#                       [-3.5678853746215704720225886381613,  -599.93433728875505658861833069295,  284.40991121755039501561856951639],
#                       [-3.5917204593487768986433648910062,  -563.28687490727614564645087114059,  285.9823971213344031422343149984],
#                       [-3.631967883344953190272001719574,  -526.04804158777455860840355228447,  287.56390165466289459555992991113],
#                       [-3.6921152434524131392252176142588,  -488.1308231582215627561427417902,  289.21758836272992120725033871054],
#                       [-3.7768714888578311994265720996644,  -449.42886920414160993597629811802,  290.99283627223038912329444727334],
#                       [-3.8928461382569553369962025990425,  -409.81042402310782617191886600949,  292.94273792410060790192272150989],
#                       [-4.0498100870811607091903985833348,  -369.10969383691770115407670971308,  295.13712864575782357512423459411],
#                       [-4.2632962516058322100616015393193,  -327.11423597167792683138021203604,  297.68223727132519040167180079948],
#                       [-4.5608300383010245917679223448538,  -283.54595513721442019694496394777,  300.76449828705264278455588922156]])

        ANS = np.array([[-3.5578006665281543679526209122965,  -636.06247287131324819053901364896,  279.70037040659677158443985334202],
                        [-3.5678853746215704720225886381613,  -599.93433728875505658861833069295,  281.39574992749890072762930515182],
                        [-3.5917204593487768986433648910062,  -563.28687490727614564645087114059,  282.96823583128290884730615672993],
                        [-3.631967883344953190272001719574,  -526.04804158777455860840355228447,  284.54974036461140030757066554656],
                        [-3.6921152434524131392252176142588,  -488.1308231582215627561427417902,  286.20342707267842691926107434597],
                        [-3.7768714888578311994265720996644,  -449.42886920414160993597629811802,  287.97867498217889482836628900486],
                        [-3.8928461382569553369962025990425,  -409.81042402310782617191886600949,  289.92857663404911361393345714532],
                        [-4.0498100870811607091903985833348,  -369.10969383691770115407670971308,  292.12296735570632928713497022954],
                        [-4.2632962516058322100616015393193,  -327.11423597167792683138021203604,  294.66807598127369611368253643491],
                        [-4.5608300383010245917679223448538,  -283.54595513721442019694496394777,  297.75033699700114848962773095309]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
     
        dw = np.array([0.00099276, 0.00235772, 0.0040782 , 0.00610312, 0.00840356,
                       0.01096123, 0.0137639 , 0.01680323, 0.02007355, 0.02357118])
        dp = np.array([0.29782801, 0.70731682, 1.22346072, 1.83093468, 2.5210683 ,
                       3.28837031, 4.12917139, 5.04096845, 6.02206468, 7.07135399])                    
        dpm = np.array([-0.08154251, -0.10789928, -0.1285909 , -0.14611083, -0.16153068,
                        -0.17542813, -0.18815679, -0.19995175, -0.21097864, -0.2213594]) 

        v_p = np.array([0.004, 1/3, 10/3]);
        v_s = np.array([0.002, 1, 1, -100/3, -100/3]);
        v_w = 1000;
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


        c_val = np.linspace(0.01, 0.08, num_pnts)

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
   
            conc = v_s[0]
            conc_ang = un.mol_lit_2_mol_angstrom( conc)

            u_p = v_p[1]
            u_a = v_s[1]
            u_b = v_s[2]

            phi_a =  conc_ang* v_w/ u_a
            phi_b =  conc_ang* v_w/ u_b
            phi_w = 1 -  phi_p -  phi_a -  phi_b
            
            F_com[ind] = (u_p / n_k * phi_p * potential_p[ind]
                         + u_a * phi_a * potential_pm[ind]
                         + u_a * phi_a * potential_pm[ind]
                         + phi_w * potential_w[ind])

        self.assertTrue(np.allclose(F_com, f_com, rtol=0.0, atol=1e-15))

        
if __name__ == '__main__':
    unittest.main()
