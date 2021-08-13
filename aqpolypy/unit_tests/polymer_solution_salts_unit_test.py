"""
:module: polymer_solution_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for the free energy calculation

.. moduleauthor:: CHI YUANCHEN <ychi@iastate.edu>, Jun 020
.. history:
"""
import unittest
import numpy as np

import aqpolypy.free_energy_polymer.PolymerSolution as Pss_2
import aqpolypy.free_energy_polymer.PolymerSolutionSalts as Pss

import aqpolypy.units.units as un

import aqpolypy.water.WaterMilleroBP as wbp

import aqpolypy.salt.SaltNaClRP as nacl  

class TestPolymerwithSalts(unittest.TestCase):
###### self.D_w need to be fixed at 55.509  to pass this test
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
        chi_s = 0.4
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
        
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])        
        
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_p(self):
        """
            checks free energy as a function of :math:`(\\phi_p)`
        """

        num_pnts = 10

        f_comp = np.array([-3.12636787182, -2.90923811928, -2.6925374651, -2.47564878314, -2.25776152297, 
                          -2.03780153794, -1.81429556898, -1.58511165737, -1.3469065977, -1.09371756816]) + \
                np.array([-6.24058591e-05, -5.70127602e-05, -5.16196612e-05, -4.62265623e-05,
                          -4.08334634e-05, -3.54403644e-05, -3.00472655e-05, -2.46541666e-05,
                          -1.92610676e-05, -1.38679687e-05])
       
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.02 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4

        #param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1, 0., 0.])
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])


        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
        
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])

            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)

            free[ind] = polymer_sol.free()
        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-11))

    def test_free_l(self):
        """
            checks free energy at large salt concentration
        """

        num_pnts = 10

        f_comp = np.array([10.8921179635, 9.93748894424, 8.97880500202, 8.01629932938, 7.05031052134, 
                           6.08131904019, 5.11002984045, 4.1375449932, 3.1657672835, 2.19856373383]) + \
                 np.array([-0.02351837, -0.02148592, -0.01945347, -0.01742102, -0.01538857,
                           -0.01335611, -0.01132366, -0.00929121, -0.00725876, -0.00522631])

        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2 , 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4

        #param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1, 0., 0.])
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])        

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free = np.zeros_like(phi_val)

        for ind, phi_p in enumerate(phi_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])
                    
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)

            free[ind] = polymer_sol.free()

        self.assertTrue(np.allclose(free, f_comp, rtol=0.0, atol=1e-8)) 

    def test_potential_w(self):
        """
            checks chemical potential as a function of :math:`(\\phi_p)`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_s = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)



        ANS = np.array([[-3.8170799226247044292964147316205, 470.10440245116910917833052963033, -369.30470445849863663639878197387], 
                        [-3.7721501186130743608622244156248, 471.74960360142472336197894122733, -347.15326919537215454025780303482], 
                        [-3.734809545151980266310988432199, 473.55067218023993616538014350681, -323.88682830708317465784329414191], 
                        [-3.708219944763620077069713604212, 475.54309572689160980768496855831, -299.40862065216084134111618197949], 
                        [-3.6968898354507042121460877459604, 477.77714197567117319215279636069, -273.69059149558987184547298454618], 
                        [-3.7075785550812143017687314006103, 480.32737300080438128807580611124, -246.71341464878268883581524661963], 
                        [-3.7511120712127175551710116241377, 483.31129442393267908739673544005, -218.43613387733744818364201023542], 
                        [-3.8464858469537436152282244838929, 486.93021220916020672109247868775, -188.74811859223366495436069456559], 
                        [-4.0318379293929636032279076318652, 491.57422873830590586557987009542, -157.34110126495032206880853031805], 
                        [-4.4027983659352579493299548185625, 498.17013582757980524884366957394, -123.24161074379232156186803059938]])

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
        chi_s = 0.4
        #param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1, 0., 0.])
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])


        for ind, phi_p in enumerate(phi_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])        
        
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w_full()
            potential_s[ind] = polymer_sol.chem_potential_s_full()
            potential_p[ind] = polymer_sol.chem_potential_p_full()
   

        dw = np.array([0.00113844, 0.00113844, 0.00113844, 0.00113844, 0.00113844,
                       0.00113844, 0.00113844, 0.00113844, 0.00113844, 0.00113844])
        ds = np.array([-0.80911723, -0.80911723, -0.80911723, -0.80911723, -0.80911723,
                       -0.80911723, -0.80911723, -0.80911723, -0.80911723, -0.80911723])
        dp = 0                
                        
                                
        self.assertTrue(np.allclose(potential_w, w_comp + dw , rtol=0.0, atol=1e-8)) 
        self.assertTrue(np.allclose(potential_s, s_comp + ds, rtol=0.0, atol=1e-8))     
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-12))

    def test_potential_w_s(self):
        """
            checks chemical potential of water as a function of :math:`(m_s)`
        """

        num_pnts = 10
        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        potential_w = np.zeros_like(phi_val)
        potential_p = np.zeros_like(phi_val)
        potential_s = np.zeros_like(phi_val)
        w_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        p_comp = np.zeros_like(phi_val)


        ANS = np.array([[- 2.9934314925524938339656058627902 , 453.4564820898114568328840512379 ,  - 442.71857140501541871824286999981], 
                        [- 3.0078462115974135404746978017521 , 455.194305669094633155780105227 ,  - 435.87963449278812940596017355688], 
                        [- 3.0231778555251397374963040487472 , 456.51613134344466788522298728026 ,  - 429.05368575495679968334616916081], 
                        [- 3.0394549540729481733065259208004 , 457.65282022231748045026922255829 ,  - 422.23995140675896593773774867486], 
                        [- 3.0567077015085815888499609638274 , 458.68718046688988476677906191981 ,  - 415.43763990979573848955075376921], 
                        [- 3.0749680874113149821671986394733 , 459.65867048426577812225701791249 ,  - 408.64594080189138097791712311846], 
                        [- 3.0942700410908515886799011185282 , 460.58933103786131646462109756612 ,  - 401.864023449886490921278969779], 
                        [- 3.1146495914323786514607550479639 , 461.49283680443728377501555470985 ,  - 395.09103571885176320321875209629], 
                        [- 3.1361450442417623388278828666653 , 462.37834743180839306569973601313 ,  - 388.32610255056783933602648772876], 
                        [- 3.1587971795046192951357186262662 , 463.25237769617703400409913783164 ,  - 381.5683244433995358196876068746]])
        for ind in range (10):
             w_comp[ind] = ANS[ind,0]
             s_comp[ind] = ANS[ind,1]
             p_comp[ind] = ANS[ind,2]
     
        dw = np.array([0.00024469, 0.0004841 , 0.00072531, 0.00095951, 0.00118238,
                       0.00139125, 0.00158426, 0.00175995, 0.0019171 , 0.00205467])
        ds = np.array([-0.50485471, -0.60297539, -0.66547724, -0.70992838, -0.74331869,
                       -0.76917484, -0.78953282, -0.80567716, -0.81847153, -0.82852529])
        dp = 0
                                       
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([2, 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4
        #param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1, 0., 0.])
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])

        c_val = np.linspace(0.1, 0.8, num_pnts)

        for ind, c_s in enumerate(c_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(c_s)
            param_s[9] = S_para.osmotic_coeff(c_s)        
        
            polymer_sol = Pss.PolymerSolutionSalts(v_p, 
                                                   np.array([c_s, 1, 1, -100/3, -100/3]), 
                                                   temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w_full()
            potential_s[ind] = polymer_sol.chem_potential_s_full()
            potential_p[ind] = polymer_sol.chem_potential_p_full()

        self.assertTrue(np.allclose(potential_w, w_comp + dw, rtol=0.0, atol=1e-8))
        self.assertTrue(np.allclose(potential_s, s_comp + ds, rtol=0.0, atol=1e-8))     
        self.assertTrue(np.allclose(potential_p, p_comp + dp, rtol=0.0, atol=1e-12)) 
        
    def test_potential_df(self):
        """
            checks if chemical potentials add up give free energy
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
        chi_s = 0.4

        #param_s = np.array([7, 7, 1, 1, 8, 8, 1, 1, 0., 0.])
        param_s = np.array([7., 7., 1., 1., 8., 8., 1., 1., 0., 0.])        
        
        obj_water_bp = wbp.WaterPropertiesFineMillero(temp )
        v_w = obj_water_bp.molar_volume() # molar volume of water 
        den = obj_water_bp.density() # density of water in SI unit from Ref[4]
        D_w = 1 / den / v_w #55.509 # mol/kg water        

        for ind, phi_p in enumerate(phi_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])  
        
            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)
            potential_w[ind] = polymer_sol.chem_potential_w_full()
            potential_p[ind] = polymer_sol.chem_potential_p_full()
            potential_s[ind] = polymer_sol.chem_potential_s_full()
            f_com[ind] = polymer_sol.free()
   
            # concentration in mols/litre
            conc = v_s[0]

            # molecular volumes
            u_p = v_p[1]
            u_a = v_s[1]
            u_b = v_s[2]
            u_s = 1/(1 / u_a + 1 / u_b)


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

    def test_no_salt(self):
        """
            checks if free energy, chemical potential, hydrogen 
            bonds returns to Dormidontova when there is no salt
        """
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([1e-12, 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4

        param_s = np.array([5., 5., 1., 1., 8., 8., 1., 1., 0., 0.])          
        num_pnts = 10

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        free_comp = np.zeros_like(phi_val) #free energy of full model
        
        w_comp = np.zeros_like(phi_val) #chemical potential  of full model
        p_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        
        free_comp_2 = np.zeros_like(phi_val) #free energy  of Dormidontova model
        
        w_comp_2 = np.zeros_like(phi_val) #chemical potential of Dormidontova model
        p_comp_2 = np.zeros_like(phi_val)
        s_comp_2 = np.zeros_like(phi_val)
        
        x_comp = np.zeros_like(phi_val) #fraction of hydrogen bonds full model
        y_comp = np.zeros_like(phi_val)       
        ha_comp = np.zeros_like(phi_val)
        hb_comp = np.zeros_like(phi_val) 
        
        x_comp_2 = np.zeros_like(phi_val) #fraction of hydrogen bonds Dormidontova model
        y_comp_2 = np.zeros_like(phi_val)    
        
        for ind, phi_p in enumerate(phi_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0])         
        
            Polymer_sol = Pss.Polymer_hydrogen_bond_shell_solver(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s)
            polymer_sol_2 = Pss_2.PolymerSolution(phi_p, x_ini, p_ini, n_k, 
                                                  v_p[1], chi_p, df_w, v_p[2])     
            
            x_comp[ind],y_comp[ind],ha_comp[ind],hb_comp[ind] = \
                Polymer_sol.solv_eqns_exact(x_ini, p_ini, param_s[0], param_s[1])

            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_comp[ind], y_comp[ind], 
                                                   n_k, chi_p, chi_s, 
                                                   np.array([ha_comp[ind], hb_comp[ind], \
                                                   1, 1, 8, 8, 1, 1, param_s[8], param_s[9]]))
            
            x_comp_2[ind] = polymer_sol_2.solv_eqns(x_ini,p_ini)[0]
            y_comp_2[ind] = polymer_sol_2.solv_eqns(x_ini,p_ini)[1]
            
            free_comp[ind] = polymer_sol.free()
            
            w_comp[ind] = polymer_sol.chem_potential_w_full()
            s_comp[ind] = polymer_sol.chem_potential_s_full()
            p_comp[ind] = polymer_sol.chem_potential_p_full()
            
            free_comp_2[ind] = polymer_sol_2.free()
            
            p_comp_2[ind] = polymer_sol_2.chem_potential()
            #s_comp_2[ind] = polymer_sol_2.chem_potential_pm()
            w_comp_2[ind] = polymer_sol_2.osm_pressure()        

        self.assertTrue(np.allclose(free_comp_2, free_comp , rtol=0.0, atol=1e-12))
        self.assertTrue(np.allclose(w_comp_2, - w_comp , rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(p_comp_2, (p_comp / n_k * v_p[1] - w_comp ), rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(x_comp, x_comp_2, rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(y_comp, y_comp_2, rtol=0.0, atol=1e-11))
        
    def test_full_chemical_potential(self):
        """
            checks if the chemical potential are correct when Eqn 301) is satisfied
        """
        v_p = np.array([0.4, 1/3, 10/3])
        v_s = np.array([0.02, 1, 1, -100/3, -100/3])
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4
        #param_s = np.array([5, 5, 1, 1, 8, 8, 1, 1, 0., 0.]) 
        param_s = np.array([5., 5., 1., 1., 8., 8., 1., 1., 0., 0.])  
                 
        num_pnts = 10

        phi_val = np.linspace(1e-1, 0.8, num_pnts)
        
        w_comp = np.zeros_like(phi_val) #fchemical potential of full model
        p_comp = np.zeros_like(phi_val)
        s_comp = np.zeros_like(phi_val)
        
        
        w_comp_2 = np.zeros_like(phi_val) #chemical potential with x,y,f solved form Eqn 301)
        p_comp_2 = np.zeros_like(phi_val)
        s_comp_2 = np.zeros_like(phi_val)
        
        x_comp = np.zeros_like(phi_val) #fraction of hydrogen bonds full model
        y_comp = np.zeros_like(phi_val)       
        ha_comp = np.zeros_like(phi_val)
        hb_comp = np.zeros_like(phi_val)         
        
        for ind, phi_p in enumerate(phi_val):
            S_para = nacl.NaClPropertiesRogersPitzer(tk=temp, pa=1)
            param_s[8] = S_para.log_gamma(v_s[0])
            param_s[9] = S_para.osmotic_coeff(v_s[0]) 
        
            Polymer_sol = Pss.Polymer_hydrogen_bond_shell_solver(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s) 
            
            x_comp[ind],y_comp[ind],ha_comp[ind],hb_comp[ind] = \
                Polymer_sol.solv_eqns_exact(x_ini, p_ini, param_s[0], param_s[1])

            polymer_sol = Pss.PolymerSolutionSalts(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_comp[ind], y_comp[ind], 
                                                   n_k, chi_p, chi_s, 
                                                   np.array([ha_comp[ind], hb_comp[ind], \
                                                   1, 1, 8, 8, 1, 1, param_s[8], param_s[9]]))        
            
            w_comp[ind] = polymer_sol.chem_potential_w_full()
            s_comp[ind] = polymer_sol.chem_potential_s_full()
            p_comp[ind] = polymer_sol.chem_potential_p_full()
            
            
            p_comp_2[ind] = polymer_sol.chem_potential_p()
            s_comp_2[ind] = polymer_sol.chem_potential_s()
            w_comp_2[ind] = polymer_sol.chem_potential_w()
      
        self.assertTrue(np.allclose(w_comp_2, w_comp , rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(p_comp_2/n_k, p_comp/n_k, rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(s_comp_2, s_comp, rtol=0.0, atol=1e-10))
        
    def test_solver_small_m(self):
        """
            checks if the solver are correct when ther is no salt
        """
        v_p = np.array([1e-12, 30/129.43, 10/3]);
        v_s = np.array([1e-12, 30/48.41,30/48.41 , 5, 5]);
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4 
        #param_s = np.array([5, 5, 1, 1, 8, 8, 1, 1])
        param_s = np.array([5., 5., 1., 1., 8., 8., 1., 1., 0., 0.])          
        
        num_puts = 10



        phi_val = np.linspace(1e-1, 0.8, num_puts)

        
        x_comp = np.zeros_like(phi_val)
        y_comp = np.zeros_like(phi_val)       
        ha_comp = np.zeros_like(phi_val)
        hb_comp = np.zeros_like(phi_val) 
        fa_comp = np.zeros_like(phi_val)
        fb_comp = np.zeros_like(phi_val) 
        
        x_comp_2 = np.zeros_like(phi_val)
        y_comp_2 = np.zeros_like(phi_val)       
        ha_comp_2 = np.zeros_like(phi_val)
        hb_comp_2 = np.zeros_like(phi_val) 

        
        x_comp_3 = np.zeros_like(phi_val)
        y_comp_3 = np.zeros_like(phi_val)       
        fa_comp_3 = np.zeros_like(phi_val)
        fb_comp_3 = np.zeros_like(phi_val)         
   
        
        x_comp_4 = np.zeros_like(phi_val)
        y_comp_4 = np.zeros_like(phi_val)       
        fa_comp_4 = np.zeros_like(phi_val)
        fb_comp_4 = np.zeros_like(phi_val)  
        
        x_comp_5 = np.zeros_like(phi_val)
        y_comp_5 = np.zeros_like(phi_val)       
        fa_comp_5 = np.zeros_like(phi_val)
        fb_comp_5 = np.zeros_like(phi_val) 
        
        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.Polymer_hydrogen_bond_shell_solver(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s) 
            
            x_comp[ind],y_comp[ind] = \
                polymer_sol.solv_eqns_xy(x_ini, p_ini)           
            
            ha_comp[ind],hb_comp[ind] = \
                polymer_sol.solv_eqns_h(param_s[0], param_s[1])   
            
            x_comp_2[ind],y_comp_2[ind],ha_comp_2[ind],hb_comp_2[ind] = \
                polymer_sol.solv_eqns_exact(x_ini, p_ini, param_s[0], param_s[1])

            x_comp_3[ind],y_comp_3[ind],fa_comp_3[ind],fb_comp_3[ind] = \
                polymer_sol.solv_eqns_small_m(x_ini, p_ini, param_s[0], param_s[1])    
            
            fa_comp[ind],fb_comp[ind] = polymer_sol.f_plus_minus([ha_comp[ind],hb_comp[ind]])
               

            x_comp_4[ind],y_comp_4[ind],fa_comp_4[ind],fb_comp_4[ind]= \
                polymer_sol.xy_firstorder(x_ini, p_ini)                      
 
        self.assertTrue(np.allclose(x_comp, x_comp_2 , rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(y_comp, y_comp_2 , rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(ha_comp, ha_comp_2 , rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(hb_comp, hb_comp_2 , rtol=0.0, atol=1e-9))     
                                                           
        self.assertTrue(np.allclose(x_comp, x_comp_3 , rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(y_comp, y_comp_3 , rtol=0.0, atol=1e-10))
        self.assertTrue(np.allclose(fa_comp, fa_comp_3 , rtol=0.0, atol=1e-9))
        self.assertTrue(np.allclose(fb_comp, fb_comp_3 , rtol=0.0, atol=1e-9))
        
    def test_solver_perturbative(self):
        """
            checks if the first and second order solver are correct 
        """ 

        con_ini = 1e-12
        con_fin = 0.02
        num_steps = 10
        con = np.linspace(con_ini, con_fin, num_steps) 
    
        v_p = np.array([1e-12, 30/129.43, 10/3]);
        v_s = np.array([1e-12, 30/48.41,30/48.41 , 5, 5]);
        temp = 298
        df_w = 10/3
        x = 0.1
        p = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4
        #param_s = np.array([5, 5, 1, 1, 8, 8, 1, 1])
        param_s = np.array([5., 5., 1., 1., 8., 8., 1., 1., 0., 0.])          


        x_sol = np.zeros_like(con) # exact
        p_sol = np.zeros_like(con)
        a_sol = np.zeros_like(con)
        b_sol = np.zeros_like(con)
        f_a = np.zeros_like(con)
        f_b = np.zeros_like(con)       
        
        X_sol = np.zeros_like(con) #perturbative
        P_sol = np.zeros_like(con)  
        F_a = np.zeros_like(con)
        F_b = np.zeros_like(con)

        
        X2_sol = np.zeros_like(con) #second order
        P2_sol = np.zeros_like(con) #second order
        FA2_sol = np.zeros_like(con) #second order
        FB2_sol = np.zeros_like(con) #second order      
          
        for ind, co in enumerate(con):

            
            
            polymer_sol = Pss.Polymer_hydrogen_bond_shell_solver(v_p, np.array([co, 30/48.41,30/48.41 , 5, 5])
                                                   , temp, df_w, x, p, 
                                                   n_k, chi_p, chi_s, param_s) 
                      
            x_sol[ind] = polymer_sol.solv_eqns_exact(x, p, param_s[0], param_s[1])[0]
            p_sol[ind] = polymer_sol.solv_eqns_exact(x, p, param_s[0], param_s[1])[1]            
            a_sol[ind] = polymer_sol.solv_eqns_exact(x, p, param_s[0], param_s[1])[2]
            b_sol[ind] = polymer_sol.solv_eqns_exact(x, p, param_s[0], param_s[1])[3]
            
            f_a[ind],f_b[ind] = polymer_sol.f_plus_minus([a_sol[ind], b_sol[ind]])
            
            if ind == 0:
                x_0 = polymer_sol.solv_eqns_small_m(x, p, param_s[0], param_s[1])[0]
                y_0 = polymer_sol.solv_eqns_small_m(x, p, param_s[0], param_s[1])[1]
            
            X_sol[ind] = x_0 + polymer_sol.xy_firstorder(x_0, y_0)[0]
            P_sol[ind] = y_0 + polymer_sol.xy_firstorder(x_0, y_0)[1]                        

            F_a[ind] = polymer_sol.xy_firstorder(x_0, y_0)[2]
            F_b[ind] = polymer_sol.xy_firstorder(x_0, y_0)[3]           

 
            X2_sol[ind], P2_sol[ind], FA2_sol[ind], FB2_sol[ind] =  \
                polymer_sol.xy_secondorder(x_0, y_0)
                
        self.assertTrue(np.allclose(f_a, F_a, rtol=0.0, atol=1e-5))
        self.assertTrue(np.allclose(f_b, F_b, rtol=0.0, atol=1e-5))
        self.assertTrue(np.allclose(X_sol, x_sol , rtol=0.0, atol=1e-6))
        self.assertTrue(np.allclose(P_sol, p_sol , rtol=0.0, atol=1e-5))     
                                                           
        self.assertTrue(np.allclose(f_a, F_a + FA2_sol, rtol=0.0, atol=1e-5))
        self.assertTrue(np.allclose(f_b, F_b + FB2_sol , rtol=0.0, atol=1e-5))
        self.assertTrue(np.allclose(X_sol + X2_sol, x_sol , rtol=0.0, atol=1e-6))
        self.assertTrue(np.allclose(P_sol + P2_sol, p_sol , rtol=0.0, atol=1e-6))     

    def test_f_2_h(self):
        """
            checks if the function converts f to h correctly
        """ 
        v_p = np.array([1e-12, 30/129.43, 10/3]);
        v_s = np.array([1e-12, 30/48.41,30/48.41 , 5, 5]);
        temp = 298
        df_w = 10/3
        x_ini = 0.1
        p_ini = 0.2
        n_k = 100
        chi_p = 0.5
        chi_s = 0.4
        #param_s = np.array([5, 5, 1, 1, 8, 8, 1, 1])
        param_s = np.array([5., 5., 1., 1., 8., 8., 1., 1., 0., 0.])          
        
        num_puts = 10

 

        #u_s = 1/(1 / v_s[1] + 1 /v_s[2])
        #nu_a = param_s[6]
        #nu_b = param_s[7]

        phi_val = np.linspace(1e-1, 0.8, num_puts)

        
        x_comp = np.zeros_like(phi_val)
        y_comp = np.zeros_like(phi_val)       
        ha_comp = np.zeros_like(phi_val)
        hb_comp = np.zeros_like(phi_val) 
        fa_comp = np.zeros_like(phi_val)
        fb_comp = np.zeros_like(phi_val) 
        
     
        ha_comp_2 = np.zeros_like(phi_val)
        hb_comp_2 = np.zeros_like(phi_val) 


        
        for ind, phi_p in enumerate(phi_val):
            polymer_sol = Pss.Polymer_hydrogen_bond_shell_solver(np.array([phi_p, 1/3, 10/3]), 
                                                   v_s, temp, df_w, x_ini, p_ini, 
                                                   n_k, chi_p, chi_s, param_s) 
            

            x_comp[ind],y_comp[ind],ha_comp[ind],hb_comp[ind] = \
                polymer_sol.solv_eqns_exact(x_ini, p_ini, param_s[0], param_s[1])  
            
            fa_comp[ind],fb_comp[ind] = polymer_sol.f_plus_minus([ha_comp[ind],hb_comp[ind]])
            
            ha_comp_2[ind],hb_comp_2[ind] = polymer_sol.h_plus_minus([fa_comp[ind],fb_comp[ind]])
       
        self.assertTrue(np.allclose(ha_comp, ha_comp_2 , rtol=0.0, atol=1e-14))
        self.assertTrue(np.allclose(hb_comp, hb_comp_2 , rtol=0.0, atol=1e-14))          
                                                                                         
if __name__ == '__main__':
    unittest.main()
