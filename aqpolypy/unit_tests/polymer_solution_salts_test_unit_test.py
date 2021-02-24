"""
:module: polymer_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: CHI YUANCHEN <ychi@iastate.edu>, January 2020
.. history:
"""
import unittest
import numpy as np

import aqpolypy.free_energy_polymer.PolymerSolutionSalts_test as Pss
import aqpolypy.free_energy_polymer.PolymerSolutionSalts_test_2 as Pss_2


#import aqpolypy.salts_theory.Bjerrum as bj
#import aqpolypy.water.WaterMilleroAW as aw

#import aqpolypy.units.units as un


class TestPolymerwithSalts(unittest.TestCase):


        
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
        
        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 0.2
        
        z_s = 1

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
            f_com[ind] = polymer_sol.free()
   
            # concentration in mols/litre
            conc = c_s
            # molecular volumes
            u_p = 1
            u_a = 1
            u_b = 1


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


    def test_potential_df_2(self):
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
        
        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 0.2
        
        z_s = 1

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss_2.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
            f_com[ind] = polymer_sol.free()
   
            # concentration in mols/litre
            conc = c_s
            # molecular volumes
            u_p = 1
            u_a = 1
            u_b = 1


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
        
    def test_free_p(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10


        f_comp = np.array([-0.316090810052, -0.334619208771, -0.351189126882, -0.365062946811, -0.375285713748, 
                           -0.380632800781, -0.379498934768, -0.369696941653, -0.348071065904, -0.309654615006]) 

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 2
        
        z_s = 1

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-12))        
        
        
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



        ANS = np.array([[-0.061458705194720352219257927979534, -83.641994723018659514679473687693, -3.4801746851007704504853655647079], 
			[-0.068340441088959455973267462930432, -80.794226856184767673538238419439, -3.4294905708736714605895656848844], 
			[-0.078376950421336581087904636644392, -77.225104007579708852160305099233, -3.386135145166877178518356067416], 
			[-0.093628397367979109083239042804481, -72.829563581365146209684047207134, -3.3522214276014423392524056721786], 
			[-0.1168986305911998439994742976511, -67.576555356729366526996605202271, -3.3306398135103971575011160000113], 
			[-0.15215883213606862777984938400744, -61.456768717936241535129693325246, -3.3254935217966725650583399709337], 
			[-0.20532313660709734323846741010922, -54.469773294598531994045731430276, -3.34289293236618360508462591274], 
			[-0.28579877662573792982195208800533, -46.619646032668164390387610662714, -3.3925373107735941285534146671932], 
			[-0.40997713763770268396221130996482, -37.91328069552618353771356041193, -3.4912637628635716628439783493221], 
			[-0.6105928690868562235525597581276, -28.359849962556338151976192962778, -3.6725259618162241550274119128439]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
         

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 2
        
        z_s = 1



        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
   

                                
        self.assertTrue(np.allclose(potential_w, w_comp , rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(potential_p, p_comp , rtol=0.0, atol=1e-13))        
        self.assertTrue(np.allclose(potential_pm, pm_comp , rtol=0.0, atol=1e-15))

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


        ANS = np.array([[-0.010665429763508390987718785561378, -82.604295577811881383725745031654, -5.940667368198243305068076414166], 
			[-0.013179824338845439656915008511118, -82.600226258490529787290035379499, -5.4252115073532344904352550662985], 
			[-0.015682563000130255423164495759414, -82.573766328700507446253231691458, -5.1065183873997205643395502772108], 
			[-0.018177321513383804770972547101837, -82.530467652998417792467589215732, -4.87660543002721756341080738828], 
			[-0.020665962290295284278578927944503, -82.473545830465341785128763341106, -4.6972911863432736718658097596624], 
			[-0.023149583505798172773225852144607, -82.405135670183822429318842850421, -4.5506399049843768767932561392087], 
			[-0.025628893383417525518678256948868, -82.32677181430338187069495337056, -4.4267953349442313278408467347447], 
			[-0.028104377164843391402566452966227, -82.239617326850480774624863045119, -4.3197637498648816085054791269471], 
			[-0.030576382393412267470737646302137, -82.144587699894583740500841884113, -4.2256308087771509033941583211025], 
			[-0.033045166698647226077091869009261, -82.042424455895265021643236735827, -4.1417006712972615864887943049055]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
     

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        phi_p = 0.2
        
        z_s = 1

        c_val = np.linspace(0.1, 0.8, num_pnts)

        for ind, c_s in enumerate(c_val):
            polymer_sol = Pss.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()

        self.assertTrue(np.allclose(potential_w, w_comp , rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(potential_p, p_comp , rtol=0.0, atol=1e-13))        
        self.assertTrue(np.allclose(potential_pm, pm_comp , rtol=0.0, atol=1e-15))        
        
    def test_free_p_2(self):
        """
            checks free energy as a function of math:`\\phi_p`
        """

        num_pnts = 10


        f_comp = np.array([-0.321502007627,-0.343245770203,-0.362201661488,-0.377669133108,-0.388731296056,
                           -0.394202972262,-0.392520254684,-0.381540027329,-0.358154480167,-0.317450764255]) 

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 2
        
        z_s = 1

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss_2.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-8))        
        
        
    def test_potential_w_2(self):
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



        ANS = np.array([[-0.053939186262791870014781715238217,  -91.013324498798209761032951270465,  -3.5953605419118620069654501669731], 
			[-0.05623508706550448376228039382152,  -86.710426097638357277944121115354,  -3.630789776047908146529904355182], 
			[-0.062711766111120545658455527338049,  -81.874803061877855281380789165269,  -3.6703997884268574482626891486436], 
			[-0.075356394410367037218784896923651,  -76.391199068258871913983387713643,  -3.7161777500594371578942695655279], 
			[-0.096897980108695737927622902707636,  -70.218394423575980482776781599341,  -3.770852669091099323777586255213], 
			[-0.13122980009804117689366058374656,  -63.336855302786277636430123383349,  -3.8383178224137781081679726691291], 
			[-0.18418321928177712436941731305451,  -55.735768618649703228830338819222,  -3.9244045749308470279670055247401], 
			[-0.26507499733395309640562702291344,  -47.408517269682255165265394314389,  -4.0384296863163565043716024538512], 
			[-0.39019347588570373676656807826468,  -38.350755753372838958471791048765,  -4.1966814982014405976690901711734], 
			[-0.59214830937593339473500042569043,  -28.559476361477378352285438900537,  -4.4317696650250035317677999024966]])

        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
         

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        c_s = 2
        
        z_s = 1



        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss_2.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()
   

                                
        self.assertTrue(np.allclose(potential_w, w_comp , rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(potential_p, p_comp , rtol=0.0, atol=1e-13))        
        self.assertTrue(np.allclose(potential_pm, pm_comp , rtol=0.0, atol=1e-15))

    def test_potential_w_s_2(self):
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


        ANS = np.array([[-0.010421945712589705725775734990179,  -82.701951998129409931309607983962,  -6.0219686054469078435131754178933], 
			 [-0.012642005157846081548348371112467,  -82.819200820738113510977029818605,  -5.5264119042078715709131672695342], 
			 [-0.014801882483594329180750679905709,  -82.935798494843988897667896864618,  -5.2219642665708839920088793484787], 
			 [-0.016917815264230493562373379076713,  -83.051750429716197504648178551179,  -5.0033406273782857115505276479617], 
			 [-0.018998699531560218052117332276918,  -83.167061974874310185251724547584,  -4.833464907812370643267960157452], 
			 [-0.021050177159903789398748491859537,  -83.281738420911010624403392199611,  -4.6949736515192934314900582992713], 
			 [-0.023076147509988855574677383414306,  -83.395785000301243194975597816665,  -4.5783486027314242908627453521042], 
			 [-0.025079465338099264874179928144771,  -83.509206888198066393025342790679,  -4.4778135674262945835301039099186], 
			 [-0.027062309454020287557968324140822,  -83.622009203215459972879930461431,  -4.3896030083387860132445798533318], 
			 [-0.029026396266080589558182360743199,  -83.734197008198340685808380712274,  -4.3111276318531528117241821695949]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             p_comp[ind] = ANS[ind,1]
             pm_comp[ind] = ANS[ind,2]
     

        A = 0.392
        b = 1.2
        #W = 18.0/1000 #kg/mol 
        D_w  = 55.509  #mol/kg

        chi_13 = 0.426
        chi_23 = 0

        g_12 = 0
        diff_g = 0

        n_k = 155
        
        phi_p = 0.2
        
        z_s = 1

        c_val = np.linspace(0.1, 0.8, num_pnts)

        for ind, c_s in enumerate(c_val):
            polymer_sol = Pss_2.PolymerSolutionSalts_test(phi_p, c_s, A, b, chi_13, chi_23, D_w, g_12, diff_g, n_k, z_s)
            potential_w[ind] = polymer_sol.chem_potential_w()
            potential_p[ind] = polymer_sol.chem_potential_p()
            potential_pm[ind] = polymer_sol.chem_potential_pm()

        self.assertTrue(np.allclose(potential_w, w_comp , rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(potential_p, p_comp , rtol=0.0, atol=1e-13))        
        self.assertTrue(np.allclose(potential_pm, pm_comp , rtol=0.0, atol=1e-15))
                

        
if __name__ == '__main__':
    unittest.main()
