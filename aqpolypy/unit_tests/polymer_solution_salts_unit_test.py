"""
:module: polymer_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: CHI YUANCHEN <ychi@iastate.edu>, Jun 020
.. history:
"""
import unittest
import numpy as np

#import aqpolypy.free_energy_polymer.PolymerSolutionSalts_full as Pss
import aqpolypy.free_energy_polymer.PolymerSolutionSalts as Pss

#import aqpolypy.salts_theory.Bjerrum as bj
#import aqpolypy.water.WaterMilleroAW as aw

import aqpolypy.units.units as un


class TestPolymerwithSalts(unittest.TestCase):

    def test_free_c(self):
        """ checks free energy when there is no salt
        """

        num_pnts = 10

        f_comp = np.array([-3.27242457671,-3.04309724043,-2.8141596695,-2.58499055405,-2.35477419179,
                           -2.12242993842,-1.8864760909,-1.64476928859,-1.39395014167,-1.12803135003])                   
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([1e-12 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])


        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p,  param_s)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_p(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10

        f_comp = np.array([-3.12639576669,-2.90928202916,-2.69259390462,-2.47571426684,-2.25783256534,
                           -2.03787465338,-1.81436727179,-1.58517846167,-1.3469650174,-1.09376411685]) +\
                 np.array([-6.24050188e-05, -5.70119951e-05, -5.16189650e-05, -4.62259433e-05,
                           -4.08329154e-05, -3.54398894e-05, -3.00468580e-05, -2.46538298e-05,
                           -1.92608096e-05, -1.38677807e-05])
       
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.02 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5

        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])



        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, param_s)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([10.8895156066,9.93339154752,8.97353790732,8.01018788335,7.04368007655,
                           6.07449495666,5.10333748772,4.13130975349,3.16031455621,2.19421894147]) + \
                 np.array([-0.02351808, -0.02148565, -0.01945322, -0.0174208 , -0.01538837,
                           -0.01335595, -0.01132352, -0.00929109, -0.00725867, -0.00522624])

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5

        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, param_s)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-8)) #-10

    def test_potential_w(self):
        """
            checks chemical potential as a function of math:`\\phi_p`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_s = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)



        ANS = np.array([[-3.8146470777637145754693573374006, 470.02890247953475982151116951968, -375.84164021191258200915719100976],
                        [-3.7682059015424449563858716460985, 471.6149032532382397928105710605, -352.60946913419070194994908717945],
                        [-3.7296792313096761036341278416906, 473.35612069999325521307831365903, -328.35988188376693810294204567413],
                        [-3.7022288014546609933945985060344, 475.28804229647657714902253545119, -302.99611750815293044480913664529],
                        [-3.6903631169590918347823971235044, 477.46093567765485672638181569027, -276.49012154492166168373579182571],
                        [-3.7008414935279721665875238278964, 479.94936275001467709810931694392, -248.82256822542987891033750003089],
                        [-3.7444898578094807497303360410701, 482.8708288276408695793850966993, -219.95250202163887665723907005599],
                        [-3.8403035883896953922318068253006, 486.42663924271030199270038352211, -189.76929368686492713425018052931],
                        [-4.0264205256801537616967445742944, 491.00689483862144484732636939839, -157.96467874860511929058226598954],
                        [-4.3984700452954488547607780901672, 497.53838244732904119715932722556, -123.56519544594005826157467886617]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             s_comp[ind] = ANS[ind,1]
             p_comp[ind] = ANS[ind,2]
         

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2, 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])



        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_s[ind] = polymer_sol.chem_potential_s()
            potential_p[ind] = polymer_sol.chem_potential_p()
   
        dw = np.array([0.00113843, 0.00113843, 0.00113843, 0.00113843, 0.00113843,
                      0.00113843, 0.00113843, 0.00113843, 0.00113843, 0.00113843])
        ds = np.array([-0.80911723, -0.80911723, -0.80911723, -0.80911723, -0.80911723,
       -0.80911723, -0.80911723, -0.80911723, -0.80911723, -0.80911723])
        dp = np.array([ 9.09494702e-13,  0.00000000e+00, -1.13686838e-13, -5.68434189e-14,
                       0.00000000e+00,  5.68434189e-13,  4.26325641e-13,  3.69482223e-13,
                       -3.69482223e-13,  1.42108547e-14])
        #dw=0; ds=0; dp=0;                
                        
                                
        self.assertTrue(np.allclose(potential_w, w_comp + dw , rtol=0.0, atol=1e-8))    #-14
        self.assertTrue(np.allclose(potential_s, s_comp + ds, rtol=0.0, atol=1e-8))     #-12    
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-8))        #-12

    def test_potential_w_s(self):
        """
            checks chemical potential of water as a function of math:`c_s`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_s = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)


        ANS = np.array([[-2.9930866019301853001195745651586,  453.13716126525205141195584701563,  -442.87378193277871425703740015933],
                       [-3.0072347509198270309733851124445,  454.87550940971207853041891056023,  -436.1547961656170515412633825747],
                       [-3.0223012783717490034422759048738,  456.19785643935192753078977645487,  -429.44813260086669441661921808873],
                       [-3.0383147005181781037620389329668,  457.33506347334313994756893606564,  -422.75302301546556197481274708139],
                       [-3.0553051981620262032390659967529,  458.3699386811380399042725741765,  -416.06868137264525443763663048458],
                       [-3.0733047474465894995554644808511,  459.34194047655812487979121172543,  -409.39430265271448139419163680941],
                       [-3.0923472642619038304153192275869,  460.27310962804551646571082912374,  -402.72906160676875832182819436866],
                       [-3.1124687640759081767036567378604,  461.17712081554325616317147051859,  -396.07211142681477386573440924167],
                       [-3.1337075392641555876289266580059,  462.0631336880310740811150793661,  -389.4225823251547721204346430568],
                       [-3.1561043563515660543140721461253,  462.93766302066071437193439663815,  -382.7795800151618048692836460134]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             s_comp[ind] = ANS[ind,1]
             p_comp[ind] = ANS[ind,2]
     
        dw = np.array([0.00024469, 0.0004841 , 0.0007253 , 0.0009595 , 0.00118236,
                       0.00139123, 0.00158424, 0.00175993, 0.00191708, 0.00205464])
        ds = np.array([-0.50485471, -0.60297539, -0.66547724, -0.70992838, -0.74331869,
                       -0.76917484, -0.78953282, -0.80567716, -0.81847153, -0.82852529])
        dp = np.array([ 5.68434189e-14, -1.70530257e-13,  1.70530257e-13, -5.68434189e-14,
                       -5.68434189e-14, -3.41060513e-13, -5.68434189e-14,  2.27373675e-13,
                       1.13686838e-13, -1.13686838e-13])
        #dw=0; ds=0; dp=0;                                        

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2, 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        #chi_e = 0.5
        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])


        c_val = np.linspace(0.1, 0.8, num_pnts)

        for ind, c_s in enumerate(c_val):
            polymer_sol = Pss.PolymerSolutionSalts(v_p, np.array([c_s, 1, 1, -100/3, -100/3]), temp, df_w, x_ini, p_ini, n_k, chi_p, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_s[ind] = polymer_sol.chem_potential_s()
            potential_p[ind] = polymer_sol.chem_potential_p()

        self.assertTrue(np.allclose(potential_w, w_comp + dw, rtol=0.0, atol=1e-8)) #-15
        self.assertTrue(np.allclose(potential_s, s_comp + ds, rtol=0.0, atol=1e-8))   #-12     
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-8))  #-12  
        
    def test_potential_df(self):
        """
            checks if chemical potentials add up give freee nergy
        """
        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_s = np.zeros_like(phi_val)
        f_com = np.zeros_like(phi_val)
        F_com = np.zeros_like(phi_val)
        
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5

        param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1])

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), v_s, temp, df_w, x_ini, p_ini, n_k, chi_p, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_s[ind] = polymer_sol.chem_potential_s()
            f_com[ind] = polymer_sol.free()
   
            # concentration in mols/litre
            conc = v_s[0]

            # molecular volumes
            u_p = v_p[1]
            u_a = v_s[1]
            u_b = v_s[2]
            u_s = 1/(1 / u_a + 1 / u_b)


            D_w = 55.509

            # volume fractions       

            V_s = conc / u_s / D_w 
            V_w = 1 
            V_all = (V_s + V_w) / (1 - phi_p)

            phi_s = V_s / V_all
            phi_w = 1-phi_s-phi_p


            
            F_com[ind] = (u_p / n_k * phi_p * potential_p[ind]
                         + u_s * phi_s * potential_s[ind]
                         + phi_w * potential_w[ind])

        self.assertTrue(np.allclose(F_com, f_com, rtol=0.0, atol=1e-14))

        
if __name__ == '__main__':
    unittest.main()
