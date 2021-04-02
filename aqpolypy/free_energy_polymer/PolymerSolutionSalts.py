"""
:module: PolymerSolutionSalts
:platform: Unix, Windows, OS
:synopsis: Defines a Polymer in Solvent, including hydrogen bonds

.. moduleauthor:: Chi Yuanchen <ychi@iastate.edu>, July2020
.. history:
..                  Alex Travesset <trvsst@ameslab.gov>, July2020
..                  - changes
..
"""
import numpy as np
from scipy.special import xlogy as lg
#from scipy.special import gamma as gamma

#import aqpolypy.units.units as un
#import aqpolypy.salts_theory.DebyeHuckel as dh
#import aqpolypy.salts_theory.HardCore as hc

from scipy.optimize import fsolve

import aqpolypy.units.concentration as con
import aqpolypy.water.WaterMilleroBP as wbp
import aqpolypy.salt.SaltNaClRP as nacl

class PolymerSolutionSalts(object):
    """
    Defines a solution with polymers with hydrogen bonds generalizing the \
    model :cite:`Dormidontova2002`
    """

    def __init__(
                 self, v_p, v_s, temp, df_w, x_ini, p_ini, n_k, chi_p,
                 param_s):

        """
        The constructor, with the following parameters

        :param v_p: polymer parameters :math:`(\\phi_p, \\frac{\\upsilon_w} \
        {\\upsilon_p}, \\Delta F_p)`
        :param v_s: salt parameters (see definition below)
        :param temp: temprature in Kelvin
        :param df_w: free energy change upon formation of hydrogen bond \
        in water (in :math:`k_BT` units)
        :param x_ini: fraction of polymer hydrogen bonds
        :param p_ini: fraction of water hydrogen bonds
        :param n_k: number of Kuhn lengths for the polymer
        :param chi_p: Flory Huggins parameter
        :param param_s: microscopic salt parameters 
        :math:`(h_+, h_-, d_+, d_-, m_+, m_-, \\nu_+, \\nu_-)`
        (number of water molecules \
        forming the hydration shell, diameter, maximum number of water \
        molecules that maybe bound to each ion,  the number of ions per salt)

        the parameter v_s is given by
        :math:`(c_s, \\frac{\\upsilon_w}{\\upsilon_+}, \
        \\frac{\\upsilon_w}{\\upsilon_-}, \\Delta F_a, \\Delta F_b)`

        where :math:`c_s` is the concentration in mol/kg
        """
        #concentration of salt (NaCl)

        # concentration in mol/kg
        self.conc_l = v_s[0]

        # concentration in mol/L

        self.T = temp

        salt_nacl = nacl.NaClPropertiesRogersPitzer(self.T)
        obj_water_bp = wbp.WaterPropertiesFineMillero(self.T)

        v_solvent = obj_water_bp.molar_volume()
        m_solvent = obj_water_bp.MolecularWeight
        v_solute = salt_nacl.molar_vol(self.conc_l)

        self.conc = con.molality_2_molarity(self.conc_l,
                                            v_solvent, 
                                            v_solute,
                                            m_solvent)  
                                                                         
                               

        # molecular volumes
        self.u_p = v_p[1]
        self.u_a = v_s[1]
        self.u_b = v_s[2]
        self.u_s = 1/(1 / self.u_a + 1 /self.u_b)

        self.D_w = 55.509 # mol/kg water

        # volume fractions
        self.phi_p = v_p[0]

        self.V_s = self.conc_l / self.u_s / self.D_w
        self.V_w = 1
        self.V_all = (self.V_s + self.V_w) / (1 - self.phi_p)

        self.phi_s = self.V_s / self.V_all
        self.phi_w = self.V_w / self.V_all

        # polymer and polymer interaction parameters
        self.n = n_k
        self.chi_p = chi_p

        # hydration numbers
        self.h_a = param_s[0]
        self.h_b = param_s[1]
        self.m_a = param_s[4]
        self.m_b = param_s[5]

        # number of ions per salt
        self.nu_a = param_s[6]
        self.nu_b = param_s[7]
        self.nu = self.nu_a + self.nu_a

        # fraction of hydration bonds (in water:x, in polymer:y)
        self.x = x_ini
        self.p = p_ini

        # fraction of water in each ion size
        self.f_a = self.h_a * self.nu_a * self.u_s * self.phi_s / self.phi_w
        self.f_b = self.h_b * self.nu_b * self.u_s * self.phi_s / self.phi_w

        # relative free energies of water association
        self.df_w = df_w
        self.df_p = v_p[2]
        self.df_a = v_s[3]
        self.df_b = v_s[4]

        self.i_size_a = param_s[2]
        self.i_size_b = param_s[3]
        self.i_size = 0.5 * (self.i_size_a + self.i_size_b)

        # salt free energy
        #self.bjerrum_object = b_o
        #self.dh_free = dh.DebyeHuckel(self.bjerrum_object)

        # hard core free energy
        #self.b_fac = b_fac
        #self.hc_free = hc.HardCore(b_fac=self.b_fac)

        # Derivative of parameters
        self.df_adp = self.f_a / self.phi_w
        self.df_ads = self.f_a * (1 / self.phi_s + 1 / self.phi_w)#

        self.df_bdp = self.f_b / self.phi_w
        self.df_bds = self.f_b * (1 / self.phi_s + 1 / self.phi_w)

        self.dxdp = - self.x / self.phi_p
        self.dxds = 0

        self.dydp = self.p / self.phi_w
        self.dyds = self.p / self.phi_w
        
        S_para = nacl.NaClPropertiesRogersPitzer(tk=self.T, pa=1)
 
        self.gamma = S_para.log_gamma(self.conc_l)
        self.osm = S_para.osmotic_coeff(self.conc_l)

    def free(self):
        """
        Free energy of a polymer in solution 
        Free energy of a polymer in solution \
        :math:`\\frac{F \\upsilon}{V k_B T}=\
        \\frac{f}{k_B T}`
        (per unit volume in :math:`k_B T` units)\
        
        :return: value of free energy (float)
        """

        f_ref_11 = (lg(self.u_p * self.phi_p / self.n,
                    self.phi_p / (self.n * np.exp(1))))

        f_ref_12 = self.nu * self.u_s * self.phi_s * \
                   (np.log(self.conc_l) - 1)
        f_ref_13 = 0
        f_ref_2 = lg(self.phi_w, self.phi_w / np.exp(1))

        f_int_1 = self.chi_p * self.phi_p * self.phi_w
        #f_int_2 = self.chi_e * self.phi_p * self.phi_1
        f_int_2 = 0
        f_as_1_1 = (lg(self.x, self.x) + lg(1 - self.x, 1 - self.x)
                    - self.x * self.df_p)
        f_as_1 = 2 * self.u_p * self.phi_p * f_as_1_1

        f_as_2_1 = (lg(self.p, self.p)
                    + lg(1 - self.f_a - self.p, 1 - self.f_a - self.p)
                    - self.p * self.df_w)
        f_as_2 = 2 * self.phi_w * f_as_2_1

        z_val = (1 - self.f_b - self.p
                 - self.x * self.u_p * self.phi_p / self.phi_w)
        f_as_3 = 2 * self.phi_w * lg(z_val, z_val)

        f_as_4_1 = self.x * self.u_p * self.phi_p / self.phi_w + self.p
        f_as_4 = (-2 * self.phi_w * lg(f_as_4_1, 2 * self.phi_w / np.exp(1)))

        f_as_5_1 =  - self.df_a * self.f_a
        f_as_5_2 =  - self.df_b * self.f_b

        f_as_5 = self.phi_w * (f_as_5_1 + f_as_5_2)
        f_as_6_1 = lg(1 - self.f_a - self.f_b, 1 - self.f_a - self.f_b)
        f_as_6_2 = (lg(1 - self.f_a, 1 - self.f_a)
                    + lg(1 - self.f_b, 1 - self.f_b))
        f_as_6 = self.phi_w * (f_as_6_1 - f_as_6_2)

        f_as_7_0 = np.log(self.phi_w / np.exp(1))
        f_as_7 = - self.phi_w * (self.f_a + self.f_b) * f_as_7_0
        
        #A_a = self.m_a * self.u_a * self.phi_a / self.phi_w
        #B_b = self.m_b * self.u_b * self.phi_b / self.phi_w
        A_a = self.nu_a * self.m_a * self.conc_l / self.D_w
        B_b = self.nu_b * self.m_b * self.conc_l / self.D_w
        
        f_as_0_1 = ((A_a - self.f_a) * np.log(1 - self.f_a / A_a) 
                    + self.f_a * np.log(self.f_a / A_a))
        f_as_0_2 = ((B_b - self.f_b) * np.log(1 - self.f_b / B_b) 
                    + self.f_b * np.log(self.f_b / B_b))                    
        f_as_0 = self.phi_w * (f_as_0_1 + f_as_0_2)

        f_ref_all = f_ref_11 + f_ref_12 + f_ref_13 + f_ref_2
        f_int_all = f_int_1 + f_int_2
        f_as_all = (f_as_1 + f_as_2 + f_as_3 + f_as_4 + f_as_5 
                    + f_as_6 + f_as_7 + f_as_0)
        
        f_exc_0 = self.nu * self.phi_w * self.conc_l / self.D_w
        f_exc_1 = 1 - self.osm + self.gamma
        f_exc = f_exc_0 * f_exc_1
        

        return f_ref_all + f_int_all + f_as_all + f_exc

    def chem_potential_w(self):
        """
        Reduced chemical potential \
        of water :math:`\\frac{\\mu_w}{k_B T}=\\frac{1}{k_B T}
        (f- \\phi_p \\frac{\\partial f}{\\partial \\phi_p}- \
        \\phi_s \\frac{\\partial f}{\\partial \\phi_s})`

        :return: value of chemical potential (float)
        """

        mu_0 = (- self.u_p * self.phi_p / self.n
               - self.nu * self.conc_l / self.D_w) #
        mu_1_1 = np.log(self.phi_w) - self.phi_w
        mu_1_2 = self.chi_p * self.phi_p * (self.phi_p + self.phi_s)#
        
        mu_1 = mu_1_1 + mu_1_2

        mu_2 = 2 * (lg(1 - self.f_a - self.p, 1 - self.f_a - self.p) +
                    lg(self.p, self.p) - self.p * self.df_w)

        z_val = (1 - self.f_b - self.p
                 - self.x * self.u_p * self.phi_p / self.phi_w)

        mu_3 = (2 * (1 - self.f_b - self.p) * np.log(z_val)
                - 2 * self.p * np.log(2 * self.phi_w / np.exp(1))
                - 2 * self.p * (1 - self.phi_w)
                + 2 * self.x * self.u_p * self.phi_p)


        mu_4 = - self.df_a * self.f_a - self.df_b * self.f_b #
        mu_5 = (lg(1 - self.f_a - self.f_b, 1 - self.f_a - self.f_b)
                - lg(1 - self.f_a, 1 - self.f_a)
                - lg(1 - self.f_b, 1 - self.f_b))

        mu_6 = - (self.f_a + self.f_b) * (np.log(self.phi_w) - self.phi_w)


        mu_7_a = - 2 * self.u_p * self.phi_p
        mu_7_b = (self.phi_p * self.dxdp
                   + self.phi_s * self.dxds)

        mu_7_c_0 = self.x / (1 - self.x)
        mu_7_c_1 = np.exp(- self.df_p)
        mu_7_c = np.log(mu_7_c_0 * mu_7_c_1 / z_val / 2 / self.phi_w)
        mu_7 = mu_7_a * mu_7_b * mu_7_c

        mu_8_a = - 2 * self.phi_w
        mu_8_b = (self.phi_p * self.dydp
                   + self.phi_s * self.dyds)

        mu_8_c_0 = (1 - self.f_a - self.p)
        mu_8_c_1 = np.exp(- self.df_w) / z_val
        mu_8_c = np.log(self.p / mu_8_c_0 * mu_8_c_1 / 2 / self.phi_w)
        mu_8 = mu_8_a * mu_8_b * mu_8_c

        mu_9_a = (self.phi_p * self.df_adp
                   + self.phi_s * self.df_ads)
        mu_9_b_1 = (1 - self.f_a) / (1 - self.f_a - self.f_b)
        mu_9_b_2 = np.exp(- self.df_a - 1)
        mu_9_b_3 = (1 - self.f_a - self.p) ** 2 * self.phi_w
        mu_9_b_4 = self.h_a / (self.m_a - self. h_a)
        mu_9_b = np.log(mu_9_b_4 * mu_9_b_1 * mu_9_b_2 / mu_9_b_3)
        mu_9 = - self.phi_w * mu_9_a * mu_9_b

        mu_10_a = (self.phi_p * self.df_bdp
                   + self.phi_s * self.df_bds)
        mu_10_b = (1 - self.f_b) / (1 - self.f_a - self.f_b)
        mu_10_c = np.exp(- self.df_b - 1)
        mu_10_d = z_val ** 2 * self.phi_w
        mu_10_e = self.h_b / (self.m_b - self. h_b)
        mu_10_f = np.log(mu_10_e * mu_10_b * mu_10_c / mu_10_d)
        mu_10 = - self.phi_w * mu_10_a * mu_10_f
        
        #mu_14 = (self.f_a * np.log(self.h_a / (self.m_a - self.h_a)) 
        #         + self.f_b * np.log(self.h_b / (self.m_b - self.h_b)))

        A_a = self.nu_a * self.m_a * self.conc_l / self.D_w
        B_b = self.nu_b * self.m_b * self.conc_l / self.D_w

        mu_11_1 = (- self.f_a * np.log(1 - self.f_a / A_a)
                + self.f_a * np.log(self.f_a / A_a))
        mu_11_2 = (- self.f_b * np.log(1 - self.f_b / B_b)
                + self.f_b * np.log(self.f_b / B_b))
        mu_11 = mu_11_1 + mu_11_2

        mu_12 = self.nu * self.conc_l / self.D_w * (1 - self.osm )

        return (mu_0 + mu_1 + mu_2 + mu_3  + mu_4
                + mu_5 + mu_6 + mu_7 + mu_8 +mu_9 
                + mu_10 + mu_11 + mu_12)

    def chem_potential_s(self):
        """
        Reduced chemical potential of salt\
        :math:`\\frac{\\mu_s}{k_B T} =
        \\frac{ \\upsilon_s}{ \\upsilon_w}\
        \\frac{ \\mu_w}{k_B T} + \
        \\frac{ \\upsilon_s}{ \\upsilon_w} \\frac{1}{k_B T}\
        \\frac{ \\partial f}{ \\partial  \\phi_s}`

        :return: value of chemical potential (float)
        """
        ### /2 ???
        
        z_val = (1 - self.f_b - self.p
                 - self.x * self.u_p * self.phi_p / self.phi_w)

        mu_0_1 = - self.u_p / self.u_s * self.phi_p / self.n
        mu_0_2 = self.nu * np.log(self.conc_l) #
        mu_0 = mu_0_1 + mu_0_2

        mu_1_1 = - 1 / self.u_s * self.phi_w #

        mu_1_2 = - self.chi_p * self.phi_w #
        mu_1 = 1 / self.u_s * self.phi_p * mu_1_2 + mu_1_1

        mu_2 = 2 / self.u_s * \
               (self.p * self.phi_w + self.x * self.u_p * self.phi_p)

        mu_3 = 1 / self.u_s  * (self.f_a + self.f_b) * self.phi_w

        mu_4_1 = (self.dxdp * self.phi_p
                  + self.dxds * (self.phi_s - 1 ))
        mu_4_2_0 = np.exp(- self.df_p) / z_val
        mu_4_2 = np.log(self.x / (1 - self.x) * mu_4_2_0 / 2 / self.phi_w)
        mu_4 = - 2 * self.u_p / self.u_s * \
               self.phi_p * (mu_4_1 * mu_4_2)

        mu_5_1 = (self.dydp * self.phi_p
                  + self.dyds * ( self.phi_s - 1))
        mu_5_2_0 = np.exp(- self.df_w) / z_val / 2 / self.phi_w
        mu_5_2 = np.log(self.p / (1 - self.f_a - self.p) * mu_5_2_0)
        mu_5 = - 2 / self.u_s * self.phi_w * (mu_5_1 * mu_5_2)

        A_a = self.nu_a * self.m_a * self.conc_l / self.D_w
        B_b = self.nu_b * self.m_b * self.conc_l / self.D_w

        mu_6_1 = (self.df_adp  * self.phi_p
                  + self.df_ads * ( self.phi_s - 1))
        mu_6_0 = np.exp(- self.df_a - 1)
        mu_6_2_0 = (1 - self.f_a) / (1 - self.f_a - self.f_b)
        mu_6_2_1 = (1 - self.f_a - self.p)
        mu_6_2_2 = self.f_a / (A_a - self.f_a)
        mu_6_2 = mu_6_2_2 * mu_6_2_0 * mu_6_0 / (mu_6_2_1) ** 2 / self.phi_w
        mu_6 = - self.phi_w / self.u_s * (mu_6_1 * np.log(mu_6_2))

        mu_7_1 = (self.df_bdp  * self.phi_p
                 + self.df_bds * (self.phi_s - 1))
        mu_7_0 = np.exp(- self.df_b - 1)
        mu_7_2_0 = (1 - self.f_b) / (1 - self.f_a - self.f_b)
        mu_7_2_1 = self.f_b / (B_b - self.f_b)
        mu_7_2 = mu_7_2_1 * mu_7_2_0 * mu_7_0 / z_val ** 2 / self.phi_w
        mu_7 = - self.phi_w / self.u_s * (mu_7_1 * np.log(mu_7_2))

        mu_8_1 = + self.m_a * self.nu_a * np.log(1 - self.f_a / A_a)
        mu_8_2 = + self.m_b * self.nu_b * np.log(1 - self.f_b / B_b)
        mu_8 = mu_8_1 + mu_8_2

        mu_9 = self.nu * self.gamma #


        return (mu_0 + mu_1 + mu_2 + mu_3 + mu_4
                + mu_5 + mu_6 + mu_7 + mu_8 + mu_9)

    # @property
    def chem_potential_p(self):
        """
        Reduced chemical potential of polymers\
        :math:`\\frac{\\mu_p}{k_B T} =
        \\frac{N \\upsilon_p}{ \\upsilon_w}\
        \\frac{ \\mu_w}{k_B T} + \
        \\frac{N \\upsilon_p}{ \\upsilon_w} \\frac{1}{k_B T}\
        \\frac{ \\partial f}{ \\partial  \\phi_p}`

        :return: value of chemical potential (float)
        """

        z_val = (1 - self.f_b - self.p
                 - self.x * self.u_p * self.phi_p / self.phi_w)

        mu_1_1_0 = - self.n / self.u_p * self.phi_w 
        mu_1_1 = np.log(self.phi_p / self.n) - self.phi_p + mu_1_1_0#

        mu_1_2_0 = self.chi_p * self.phi_w
        mu_1_2 = self.n / self.u_p * (1 - self.phi_p) * (mu_1_2_0)
        mu_1 = mu_1_1 + mu_1_2

        mu_2_0 = (lg(1 - self.x, 1 - self.x)
                  + lg(self.x, self.x) - self.x * self.df_p)
        mu_2 = 2 * self.n * (mu_2_0)

        mu_3 = - 2 * self.n * self.x * (np.log(z_val) + 1)

        mu_4 = self.n / self.u_p * (self.f_a + self.f_b) * self.phi_w

        mu_5_0 = self.u_p * (np.log(2 * self.phi_w / np.exp(1)) - self.phi_p)
        mu_5 = -2 * self.n / self.u_p * (self.x * mu_5_0 - self.p * self.phi_w)

        mu_6_0 = ((self.phi_p - 1) * self.dxdp
                  + self.phi_s * self.dxds)
        mu_6_1 = self.x / (1 - self.x) * np.exp(- self.df_p) / z_val
        mu_6_2 = np.log(mu_6_1 / 2 / self.phi_w)
        mu_6 = - 2 * self.n * self.phi_p * (mu_6_0) * mu_6_2

        mu_7_0 = ((self.phi_p - 1) * self.dydp
                  + self.phi_s * self.dyds)
        mu_7_1 = self.p / (1 - self.f_a - self.p) * np.exp(- self.df_w) / z_val
        mu_7_2 = np.log(mu_7_1 / 2 / self.phi_w)
        mu_7 = - 2 * self.n / self.u_p * self.phi_w * (mu_7_0) * mu_7_2

        mu_8_0 = np.exp(- self.df_a - 1)
        mu_8_1 = ((self.phi_p - 1) * self.df_adp
                  + self.phi_s * self.df_ads)
        mu_8_2_0 = (1 - self.f_a) / (1 - self.f_a - self.f_b)
        mu_8_2_1 = self.h_a / (self.m_a - self. h_a)
        mu_8_3 = mu_8_2_1 * mu_8_2_0 * mu_8_0
        mu_8_2 = mu_8_3 / (1 - self.f_a - self.p) ** 2 / self.phi_w
        mu_8 = - self.n / self.u_p * self.phi_w * (mu_8_1) * np.log(mu_8_2)

        mu_9_0 = np.exp(- self.df_b - 1)
        mu_9_1 = ((self.phi_p - 1) * self.df_bdp
                  + self.phi_s * self.df_bds)
        mu_9_2_0 = (1 - self.f_b) / (1 - self.f_a - self.f_b)
        mu_9_2_1 = self.h_b / (self.m_b - self. h_b)
        mu_9_2 = mu_9_2_1 * mu_9_2_0 * mu_9_0 / z_val ** 2 / self.phi_w
        mu_9 = - self.n / self.u_p * self.phi_w * (mu_9_1) * np.log(mu_9_2)

        mu_10 = 0
        
        mu_t = mu_1_1

        return (mu_1 + mu_2 + mu_3 + mu_4 + mu_5 + mu_6 + mu_7
                + mu_8 + mu_9 + mu_10)

    def eqns(self, val):
        """
        Equations determining the fraction of hydrogen bonds

        :param val: ndarray containing x,p
        :return: equations (ndarray)
        """

        x, p = val

        fac = 2 * self.phi_w * (1 - p - x * self.u_p * self.phi_p / self.phi_w)

        eqn1 = np.exp(self.df_p) * (1 - x) * fac
        eqn2 = np.exp(self.df_w) * (1 - p) * fac
        
        return np.array([x - eqn1, p - eqn2])

    def solv_eqns(self, x, p):
        """
        Solution to the equations defining the fraction of hydrogen bonds

        :param x: initial value for fraction of polymer hydrogen bonds
        :param p: initial value for fraction of water hydrogen bonds
        :return: number of hydrogen bonds ( OptimizeResultsObject_ )

        .. _OptimizeResultsObject: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """

        sol = fsolve(self.eqns, np.array([x, p]))

        return sol                


    def eqns_f(self, val):
        """
        Equations determining the fraction of hydrogen bonds

        :param val: ndarray containing :math:`(h_+, h_-)`
        :return: equations (ndarray)
        """

        h_a, h_b = val
        
        x, p = self.solv_eqns(self.x, self.p)
       
        eqn1 = np.exp(self.df_a) * self.phi_w * (1 - p) ** 2 
        eqn2 = np.exp(self.df_b) * self.phi_w * (1 - x * self.u_p * self.phi_p / self.phi_w - p) ** 2 
        
        eqn1_f = eqn1 * (self.m_a - h_a)
        eqn2_f = eqn2 * (self.m_b - h_b)

        return np.array([h_a - eqn1_f, h_b - eqn2_f])

    def solv_eqns_f(self, h_a, h_b):
        """
        Solution to the equations defining the fraction of hydrogen bonds

        :param h_a: initial value for :math:`(h_+)`
        :param h_b: initial value for :math:`(h_-)`
        :return: number of hydrogen bonds ( OptimizeResultsObject_ )

        .. _OptimizeResultsObject: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """

        sol = fsolve(self.eqns_f, np.array([h_a, h_b]))

        return sol
        
    def eqns_f2(self, val): 
        """
        Equations determining the fraction of hydrogen bonds

        :param val: ndarray containing :math:`(x, p, h_+, h_-)`
        :return: equations (ndarray)        
        """
        x, p, h_a, h_b = val

        f_a = h_a * self.u_a * self.phi_a / self.phi_w
        f_b = h_b * self.u_b * self.phi_b / self.phi_w

        fac = 2 * self.phi_w * (1 - p  - x * self.u_p * self.phi_p / self.phi_w - f_b)

        eqn1 = np.exp(self.df_p) * (1 - x) * fac
        eqn2 = np.exp(self.df_w) * (1 - p - f_a) * fac

        eqn3_1 = np.exp(self.df_a) * self.phi_w * (1 - p - f_a) ** 2
        eqn4_1 = np.exp(self.df_b) * self.phi_w * (1 - x * self.u_p * self.phi_p / self.phi_w - p - f_b) ** 2
        
        eqn3 = eqn3_1 * (self.m_a - h_a) * (1 - f_a - f_b)
        eqn4 = eqn4_1 * (self.m_b - h_b) * (1 - f_a - f_b)

        return np.array([x - eqn1, p - eqn2, (1 - f_a) * h_a - eqn3, (1 - f_b) * h_b - eqn4])
        
    def solv_eqns_f2(self, x, p, h_a, h_b):
        """
        Solution to the equations defining the fraction of hydrogen bonds
        
        :param x: initial value for fraction of polymer hydrogen bonds
        :param p: initial value for fraction of water hydrogen bonds
        :param h_a: initial value for :math:`(h_+)`
        :param h_b: initial value for :math:`(h_-)`
        :return: number of hydrogen bonds ( OptimizeResultsObject_ )

        .. _OptimizeResultsObject: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """

        sol = fsolve(self.eqns_f2, np.array([x, p, h_a, h_b]))

        return sol        
        
            

