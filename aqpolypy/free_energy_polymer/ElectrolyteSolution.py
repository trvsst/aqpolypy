"""
:module: ElectrolyteSolution
:platform: Unix, Windows, OS
:synopsis: Defines an electrolyte in Solvent

.. moduleauthor::  Alex Travesset <trvsst@ameslab.gov>, April2023
.. history:
..
..
"""
import numpy as np
from scipy.special import xlogy as lg

from scipy.optimize import fsolve
import aqpolypy.units.units as un
import aqpolypy.water.WaterMilleroBP as fm

class ElectrolyteSolution(object):
    """
    Class defining an electrolyte solution described by the mean field model
    """

    def __init__(self, ml, temp, param_w, param_salt, param_h, press=1.01325, b_param=0, k_r=1e-12):

        """
        The constructor, with the following parameters

        :param ml: salt molality
        :param temp: temperature in Kelvin
        :param param_w: water parameters (see definition below)
        :param param_salt: salt parameters (see definition below)
        :param param_h: hydration layer parameters, see below
        :param press: pressure in bars, default is 1 atm
        :param b_param: b-parameter for debye huckel contribution
        :param k_r: reference compressibility

        number density must be given in the same units as the molar volume
        Energy is given in units of temperature, entropies in units of :math:`k_{B}`

        the parameters param_w is a dictionary with
        :math:`v\_w = \\upsilon_w, de\\_w = \\Delta E_w, ds\\_w = \\Delta S_w, de\\_2d=\\Delta E_{2d}, \
        ds\\_2d = \\Delta S_{2d}, de\\_2a=\\Delta E_{2a}, ds\\_2a = \\Delta S_{2a}`

        the parameters param_salt is a dictionary given by
        :math:`de\\_p0 = \\Delta E_{(+,0)}, ds\\_p0=\\Delta S_{(+,0)}, de\\_p1 = \\Delta E_{(+,1)},  \
        ds\\_p1=\\Delta S_{(+,1)}, de\\_p2 = \\Delta E_{(+,2)}, ds\\_p2=\\Delta S_{(+,2)}, de\\_m0=\\Delta E_{(-,0)}, \
        ds\\_m0=\\Delta S_{(-,0)}, de\\_m1 = \\Delta E_{(-,1)}, ds\\_m1=\\Delta S_{(-,1)}, de\\_m2 = \\Delta E_{(-,2)},\
        ds\\_m2=\\Delta S_{(-,2)}, de\\_b = \\Delta E_{B}, ds\\_b = \\Delta S_{B}, m\\_p = M_{+}, m\\_m = M_{-}, \
        mb\\_p = M_{+}^B, mb\\_m = M_{-}^B`

        the parameters param_h is a dictionary with
        :math:`h\\_p0 = h_{(+,0)}, h\\_p1 = h_{(+,1)}, h\\_p2 = h_{(+,2)}, \
        h\\_m0 = h_{(-,0)}, h\\_m1 = h_{(-,1)}, h\\_m2 = h_{(-,2)}, \
        hb\\_p0 = h^B_{(+,0)}, hb\\_p1 = h^B_{(+,1)}, h^B\\_p2 = h^B_{(+,2)}, \
        hb\\_m0 = h^B_{(-,0)}, hb\\_m1 = h^B_{(-,1)}, h^B\\_m2 = h^B_{(-,2)}`
        """

        # molality
        self.ml = ml

        # constants
        self.delta_w = un.delta_w()

        # temperature
        self.tp = temp

        # electrostatic parameter for debye-huckel
        self.b_param = b_param

        # reference compressibility
        self.k_ref = k_r

        # water model at the given temperature and pressure
        self.press = press
        wfm = fm.WaterPropertiesFineMillero(self.tp, self.press)
        self.a_gamma = 3*wfm.a_phi()

        # molecular volume
        self.u_w = param_w['v_w']
        self.u_s = param_salt['v_s']
        self.u_b = param_salt['v_b']

        # concentration in dimensionless units
        self.r_h = self.ml / self.delta_w
        self.n_w = 1/(1+self.u_s*self.r_h/self.u_w)
        self.n_s = self.u_w*self.r_h/(self.r_h*self.u_s+self.u_w)

        # square ionic strength
        self.sqrt_i_str = np.sqrt(self.ml)

        # pressure
        self.pvt = self.press*self.u_w/(un.k_bolzmann_bar_angstrom3()*self.tp)

        # energies and entropies
        self.e_w = param_w['de_w']
        self.s_w = param_w['ds_w']
        self.e_2d = param_w['de_2d']
        self.s_2d = param_w['ds_2d']
        self.e_2a = param_w['de_2a']
        self.s_2a = param_w['ds_2a']

        self.e_p = param_salt['de_p0']
        self.s_p = param_salt['ds_p0']
        self.e_p1 = param_salt['de_p1']
        self.s_p1 = param_salt['ds_p1']
        self.e_p2 = param_salt['de_p2']
        self.s_p2 = param_salt['ds_p2']
        self.e_bp = param_salt['de_bp0']
        self.s_bp = param_salt['ds_bp0']
        self.e_bp1 = param_salt['de_bp1']
        self.s_bp1 = param_salt['ds_bp1']
        self.e_bp2 = param_salt['de_bp2']
        self.s_bp2 = param_salt['ds_bp2']
        self.e_m = param_salt['de_m0']
        self.s_m = param_salt['ds_m0']
        self.e_m1 = param_salt['de_m1']
        self.s_m1 = param_salt['ds_m1']
        self.e_m2 = param_salt['de_m2']
        self.s_m2 = param_salt['ds_m2']
        self.e_bm = param_salt['de_bm0']
        self.s_bm = param_salt['ds_bm0']
        self.e_bm1 = param_salt['de_bm1']
        self.s_bm1 = param_salt['ds_bm1']
        self.e_bm2 = param_salt['de_bm2']
        self.s_bm2 = param_salt['ds_bm2']
        self.e_b = param_salt['de_b']
        self.s_b = param_salt['ds_b']

        # # hydration parameters
        self.m_p = param_h['m_p']
        self.m_m = param_h['m_m']
        self.m_bp = param_h['mb_p']
        self.m_bm = param_h['mb_m']
        self.h_p0 = param_h['h_p0']
        self.h_p1 = param_h['h_p1']
        self.h_p2 = param_h['h_p2']
        self.h_m0 = param_h['h_m0']
        self.h_m1 = param_h['h_m1']
        self.h_m2 = param_h['h_m2']
        self.hb_p0 = param_h['hb_p0']
        self.hb_p1 = param_h['hb_p1']
        self.hb_p2 = param_h['hb_p2']
        self.hb_m0 = param_h['hb_m0']
        self.hb_m1 = param_h['hb_m1']
        self.hb_m2 = param_h['hb_m2']

        # define free energies
        self.f_w = self.e_w/self.tp - self.s_w
        self.f_2d = self.e_2d/self.tp - self.s_2d
        self.f_2a = self.e_2a/self.tp - self.s_2a
        self.f_bj = self.e_b / self.tp - self.s_b
        self.f_p = self.e_p/self.tp - self.s_p
        self.f_p1 = self.e_p1/self.tp - self.s_p1
        self.f_p2 = self.e_p2/self.tp - self.s_p2
        self.f_m = self.e_m/self.tp - self.m_p
        self.f_m1 = self.e_m1/self.tp - self.s_m1
        self.f_m2 = self.e_m2/self.tp - self.s_m2
        self.f_bp = self.e_bp/self.tp - self.s_bp
        self.f_bp1 = self.e_bp1/self.tp - self.s_bp1
        self.f_bp2 = self.e_bp2/self.tp - self.s_bp2
        self.f_bm = self.e_bm/self.tp - self.s_bm
        self.f_bm1 = self.e_bm1/self.tp - self.s_bm1
        self.f_bm2 = self.e_bm2/self.tp - self.s_bm2

        # define new hydration parameters
        self.h_p = self.h_p0+self.h_p1+self.h_p2
        self.h_m = self.h_m0 + self.h_m1 + self.h_m2
        self.hb_p = self.hb_p0 + self.hb_p1 + self.hb_p2
        self.hb_m = self.hb_m0 + self.hb_m1 + self.hb_m2

    def f_assoc(self, in_p):
        """
        Defines the association free energy

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        s_hp = np.sum(in_p[3:6], axis=0)
        s_hm = np.sum(in_p[6:9], axis=0)
        s_bp = np.sum(in_p[9:12], axis=0)
        s_bm = np.sum(in_p[12:15], axis=0)

        t_0 = - self.n_w*(2*in_p[0]*self.f_w+in_p[2]*self.f_2d+in_p[1]*self.f_2a)-self.n_s*in_p[15]*self.f_bj

        t_1_0 = - self.n_s*(s_hm*self.f_m+s_hp*self.f_p)*(1-in_p[15])
        t_1_1 = - self.n_s*(s_bm*self.f_bm+s_bp*self.f_bp)*in_p[15]
        t_1 = t_1_0 + t_1_1

        t_2_0 = - self.n_s*(in_p[7]*self.f_m1+in_p[4]*self.f_p1)*(1-in_p[15])
        t_2_1 = - self.n_s*(in_p[8]*self.f_m2+in_p[5]*self.f_p2)*(1-in_p[15])
        t_2_2 = - self.n_s*(in_p[13]*self.f_bm1+in_p[10]*self.f_bp1)*in_p[15]
        t_2_3 = - self.n_s*(in_p[14]*self.f_bm2+in_p[11]*self.f_bp2)*in_p[15]
        t_2 = t_2_0+t_2_1+t_2_2+t_2_3

        va0 = (1-2*in_p[0]+in_p[1])*self.n_w-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.n_s
        t_3 = lg(va0, va0)

        va1 = 2*(in_p[0]-in_p[1])*self.n_w-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.n_s
        t_4 = lg(va1, va1)-2*(in_p[0]-in_p[1])*self.n_w*np.log(2)

        va2 = in_p[1]*self.n_w-((1-in_p[15])*in_p[5]+in_p[15]*in_p[11])*self.n_s
        t_5 = lg(va2, va2)

        vah = lg(in_p[3], in_p[3])+lg(in_p[4], in_p[4])+lg(in_p[5], in_p[5])
        t_6 = (1-in_p[15])*self.n_s*(vah-lg(s_hp, s_hp))

        vabh = lg(in_p[9], in_p[9]) + lg(in_p[10], in_p[10])+ lg(in_p[11], in_p[11])
        t_7 = in_p[15]*self.n_s*(vabh-lg(s_bp, s_bp))

        vd0 = (1-2*in_p[0]+in_p[2])*self.n_w-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.n_s
        t_8 = lg(vd0, vd0)

        vd1 = 2*(in_p[0]-in_p[2])*self.n_w-((1-in_p[15])*in_p[7]+ in_p[15]*in_p[13])*self.n_s
        t_9 = lg(vd1, vd1)-2*(in_p[0]-in_p[2])*self.n_w * np.log(2)

        vd2 = in_p[1]*self.n_w-((1-in_p[15])*in_p[8]+in_p[15]*in_p[14])*self.n_s
        t_10 = lg(vd2, vd2)

        vdh = lg(in_p[6], in_p[6])+lg(in_p[7], in_p[7])+lg(in_p[8], in_p[8])
        t_11 = (1-in_p[15])*self.n_s*(vdh - lg(s_hm, s_hm))

        vdbh = lg(in_p[12], in_p[12]) + lg(in_p[13], in_p[13]) + lg(in_p[14], in_p[14])
        t_12 = in_p[15]*self.n_s*(vdbh - lg(s_bm, s_bm))

        t_13_1 = self.n_s*self.m_p*(1-in_p[15])*lg(1-s_hp/self.m_p, 1-s_hp/self.m_p)
        t_13_2 = self.n_s*self.m_p*(1-in_p[15])*lg(s_hp/self.m_p, s_hp/self.m_p)
        t_13_3 = self.n_s*self.m_bp*in_p[15]*lg(1-s_bp/self.m_bp, 1-s_bp/self.m_bp)
        t_13_4 = self.n_s*self.m_bp*in_p[15]*lg(s_bp/self.m_bp, s_bp/self.m_bp)
        t_13 = t_13_1+t_13_2+t_13_3+t_13_4

        t_14_1 = self.n_s*self.m_m*(1-in_p[15])*lg(1-s_hm/self.m_m, 1-s_hm/self.m_m)
        t_14_2 = self.n_s*self.m_m*(1-in_p[15])*lg(s_hm/self.m_m, s_hm/self.m_m)
        t_14_3 = self.n_s*self.m_bm*in_p[15]*lg(1-s_bm /self.m_bm,1-s_bm/self.m_bm)
        t_14_4 = self.n_s*self.m_bp*in_p[15]*lg(s_bm/self.m_bm, s_bm/self.m_bm)
        t_14 = t_14_1 + t_14_2 + t_14_3 + t_14_4

        t_15_1 = self.n_s*in_p[15]*np.log(self.m_bm*self.m_bp)-2*lg(in_p[0]*self.n_w, in_p[0]*self.n_w)
        t_15_2 = 2*self.n_s*(lg(1-in_p[15], 1-in_p[15])+lg(in_p[15], in_p[15]))-lg(self.n_s*in_p[15], self.n_s*in_p[15])
        t_15 = t_15_1+t_15_2

        t_16_1 = -2*lg(self.n_w, self.n_w)-2*in_p[0]*self.n_w*(np.log(2)-1)
        t_16_2 = self.n_s*(in_p[15]*(1+s_bp+s_bm)+(1-in_p[15])*(s_hp+s_hm))
        t_16 = t_16_1 + t_16_2

        t_s = t_0+t_1+t_2+t_3+t_4+t_5+t_6+t_7+t_8+t_9+t_10+t_11+t_12+t_13+t_14+t_15+t_16

        return t_s

    def f_ideal(self):
        """
        Defines the ideal free energy

        """

        n_w = self.n_w
        n_s = self.n_s

        t_s = lg(n_w, n_w) + 2*lg(n_s, n_s) - (n_w+2*n_s)

        return t_s

    def f_comp(self, in_p):
        """
        Defines the compressible free energy

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        k_param = self.u_w/(self.tp*self.k_ref)
        v0 = self.n_w+((1-in_p[15])*self.u_s+in_p[15]*self.u_b)*self.n_s/self.u_w

        return 0.5*k_param*(1-v0)**2/v0

    def f_debye(self, in_p):
        """
        Defines the Debye-Huckel contribution

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        i_qstr = np.sqrt(1-in_p[15])*self.sqrt_i_str
        val = self.b_param*i_qstr

        return -4*self.a_gamma*i_qstr**3*self.tau_debye(val)*self.n_w/(3*self.delta_w)

    def f_total(self, in_p):
        """
        Defines the total free energy

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        f_i = self.f_ideal()
        f_a = self.f_assoc(in_p)
        f_c = self.f_comp(in_p)
        f_d = self.f_debye(in_p)

        return f_i + f_a + f_c + f_d

    def mu_w_1(self, in_p):
        """
        Defines partial contribution to the water chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_11 = (1-2*in_p[0])*np.log(self.n_w)
        t_12 = (1-2*in_p[0]+in_p[1])*np.log(1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h)
        t_1 = t_11+t_12

        t_2 = 2*(in_p[0]-in_p[1])*np.log(2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h)

        t_3 = in_p[1]*np.log(in_p[1]-((1-in_p[15])*in_p[5]+in_p[15]*in_p[11])*self.r_h)

        t_4 = (1-2*in_p[0]+in_p[2])*np.log(1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h)

        t_5 = 2*(in_p[0]-in_p[2])*np.log(2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h)

        t_6 = in_p[2]*np.log(in_p[2]-((1-in_p[15])*in_p[8]+in_p[15]*in_p[14])*self.r_h)

        t_7_1 = -2*in_p[0]*self.f_w-in_p[2]*self.f_2d-in_p[1]*self.f_2a
        t_7_2 = -2*(3*in_p[0]-in_p[1]-in_p[2])*np.log(2)-2*lg(in_p[0], in_p[0])
        t_7 = t_7_1 + t_7_2

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7

    def mu_w_debye(self, in_p):
        """
        Defines the Electrostatic chemical potential

        :param b_g: chemical potential constant
        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        x_val = np.sqrt((1-in_p[15]))*self.sqrt_i_str
        return 2*self.a_gamma*x_val**3*self.r_debye(self.b_param*x_val)/(3*self.delta_w)

    def mu_w_comp(self, in_p):
        """
        Defines the compressibility chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        s_hp = np.sum(in_p[3:9], axis=0)
        s_bp = np.sum(in_p[9:15], axis=0)
        t_1_1 = self.pvt-(1-2*in_p[0])*self.n_w-(2-in_p[15])*self.n_s
        t_1_2 = ((1-in_p[15])*s_hp+in_p[15]*s_bp)*self.n_s

        t_1 = t_1_1 + t_1_2

        return t_1

    def mu_sf_1(self, in_p):
        """
        Defines partial contribution to the free salt chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        s_hp = np.sum(in_p[3:6], axis=0)
        s_hm = np.sum(in_p[6:9], axis=0)

        t_11 = 2*np.log((1-in_p[15])*self.n_s)-s_hp*self.f_p-s_hm*self.f_m
        t_12 = in_p[4]*self.f_p1+in_p[5]*self.f_p2+in_p[7]*self.f_m1+in_p[8]*self.f_m2
        t_1 = t_11 + t_12

        t_2 = -in_p[3]*np.log(1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h)

        t_3 = -in_p[4]*np.log(2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h)

        t_4 = -in_p[5]*np.log(in_p[1]-((1-in_p[15])*in_p[5]+ in_p[15]*in_p[11])*self.r_h)

        t_5 = lg(in_p[3],in_p[3])+lg(in_p[4], in_p[4])+lg(in_p[5], in_p[5])-lg(s_hp, s_hp)

        t_6 = -in_p[6]*np.log(1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+ in_p[15]*in_p[12])*self.r_h)

        t_7 = -in_p[7]*np.log(2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+ in_p[15]*in_p[13])*self.r_h)

        t_8 = -in_p[8]*np.log(in_p[2]-((1-in_p[15])*in_p[8] + in_p[15]*in_p[14])*self.r_h)

        t_9 = lg(in_p[6], in_p[6]) + lg(in_p[7], in_p[7]) + lg(in_p[8], in_p[8]) - lg(s_hm, s_hm)

        t_10_1 = self.m_p*(lg(1-s_hp/self.m_p,1-s_hp/self.m_p)+ lg(s_hp/self.m_p,s_hp/self.m_p))
        t_10_2 = self.m_m*(lg(1-s_hm/self.m_m,1-s_hm/self.m_m)+ lg(s_hm/self.m_m,s_hm/self.m_m))
        t_10 = t_10_1 + t_10_2

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10

    def mu_sf_debye(self, in_p):
        """
        Defines the salt Electrostatic chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        x_val = np.sqrt(1-in_p[15])*self.sqrt_i_str

        return -2*self.a_gamma*x_val/(1+self.b_param*x_val)

    def mu_sb_1(self, in_p):
        """
        Defines partial contribution to the bjerrum salt chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """
        s_bp = np.sum(in_p[9:12], axis=0)
        s_bm = np.sum(in_p[12:15], axis=0)

        t_11 = np.log(in_p[15]*self.n_s)-s_bp*self.f_bp-s_bm*self.f_bm-self.f_bj
        t_12 = in_p[10]*self.f_bp1+in_p[11]*self.f_bp2+in_p[13]*self.f_bm1+in_p[14]*self.f_bm2
        t_1 = t_11 + t_12

        t_2 = -in_p[9]*np.log(1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h)

        t_3 = -in_p[10]*np.log(2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h)

        t_4 = -in_p[11]*np.log(in_p[1]-((1-in_p[15])*in_p[5]+ in_p[15]*in_p[11])*self.r_h)

        t_5 = lg(in_p[9], in_p[9])+lg(in_p[10], in_p[10])+lg(in_p[11], in_p[11]) - lg(s_bp, s_bp)

        t_6 = -in_p[12]*np.log(1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h)

        t_7 = -in_p[13]*np.log(2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h)

        t_8 = -in_p[14]*np.log(in_p[2]-((1-in_p[15])*in_p[8]+ in_p[15]*in_p[14])*self.r_h)

        t_9 = lg(in_p[12], in_p[12])+lg(in_p[13], in_p[13])+lg(in_p[14], in_p[14])-lg(s_bm, s_bm)

        t_10_1 = self.m_bp*(lg(1-s_bp/self.m_bp, 1-s_bp/self.m_bp)+lg(s_bp/self.m_bp, s_bp/self.m_bp))
        t_10_2 = self.m_bm*(lg(1-s_bm/self.m_bm, 1-s_bm/self.m_bm)+lg(s_bm/self.m_bm, s_bm/ self.m_bm))
        t_10 = t_10_1 + t_10_2

        t_11 = np.log(self.m_m*self.m_p)

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10 + t_11

    def mu_w(self, in_p):
        """
        Defines the water chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        m_1 = self.mu_w_1(in_p)
        m_2 = self.mu_w_debye(in_p)
        m_3 = self.mu_w_comp(in_p)

        m_total = m_1+m_2+m_3

        return m_total

    def mu_w0(self):
        """
        Returns the water chemical potential for pure water
        """

        in_p = np.zeros(16)
        in_p[:3] = self.solve_eqns_water_analytical()

        m_1 = self.mu_w_1(in_p)-(1-2*in_p[0])*np.log(self.n_w)
        m_2 = 0
        m_3 = self.pvt-(1-2*in_p[0])

        m_total = m_1 + m_2 + m_3

        return m_total

    def mu_sf(self, in_p):
        """
        Defines the free salt chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        m_1 = self.mu_sf_1(in_p)
        m_2 = self.mu_sf_debye(in_p)
        m_3 = self.mu_w_comp(in_p)*self.u_s/self.u_w

        m_total = m_1 + m_2 + m_3

        return m_total

    def mu_bf(self, in_p):
        """
        Defines the bjerrum salt chemical potential

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        m_1 = self.mu_sb_1(in_p)
        m_2 = self.mu_w_comp(in_p)*self.u_b/self.u_w

        m_total = m_1 + m_2

        return m_total

    def c_gamma(self, in_p):
        """
        Defines the activity coefficient

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_ideal = 2*np.log(self.ml) + self.pvt*(self.u_s+self.u_b)/self.u_w
        val = (1-in_p[15])*self.mu_sf(in_p)+in_p[15]*self.mu_sb_1(in_p)-t_ideal

        return 0.5*val

    def c_osmotic(self, in_p):
        """
        Defines the osmotic coefficient

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_ideal = self.mu_w0()
        val = -(self.mu_w(in_p)-t_ideal)*self.delta_w/self.ml

        return 0.5 * val

    def eqn_y(self, in_p):
        """
        Equation determining the number of water hydrogen bonds

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = 8*in_p[0]*np.exp(self.f_w)
        t_1_u = 2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h
        t_1_d = 1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h
        t_2_u = 2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h
        t_2_d = 1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h

        return t_0 - t_1_u*t_2_u/(t_1_d*t_2_d)

    def eqn_za(self, in_p):
        """
        Equation determining the number of water molecules with 2 acceptor hydrogen bonds

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = 0.25*np.exp(self.f_2a)
        t_1_u = 1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h
        t_2_u = in_p[1]- ((1-in_p[15])*in_p[5]+in_p[15]*in_p[11])*self.r_h
        t_1_d = 2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h

        return t_0 - t_1_u*t_2_u/t_1_d**2

    def eqn_zd(self, in_p):
        """
        Equation determining the number of water molecules with 2 donor hydrogen bonds

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        f_m = 1

        t_0 = 0.25 * np.exp(self.f_2d)
        t_1_u = 1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h
        t_2_u = in_p[2]-((1-in_p[15])*in_p[8]+in_p[15]*in_p[14])*self.r_h
        t_1_d = 2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h

        return t_0-t_1_u*t_2_u/t_1_d**2

    def eqn_h_p0(self, in_p):
        """
        Equation determining the hydration shell parameter h_p0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p)*(self.m_p-in_p[3]-in_p[4]-in_p[5])
        t_1_u = in_p[3]
        t_1_d = 1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_p0(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp) * (self.m_bp - in_p[9] - in_p[10] - in_p[11])
        t_1_u = in_p[9]
        t_1_d = 1-2*in_p[0]+in_p[1]-((1-in_p[15])*in_p[3]+in_p[15]*in_p[9])*self.r_h

        return self.n_w*t_0 - t_1_u / t_1_d

    def eqn_h_p1(self, in_p):
        """
        Equation determining the hydration shell parameter h_p1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p+self.f_p1)*(self.m_p-in_p[3]-in_p[4]-in_p[5])
        t_1_u = in_p[4]
        t_1_d = 2*(in_p[0]-in_p[1])-((1-in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_p1(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp + self.f_bp1) * (self.m_bp - in_p[9] - in_p[10] - in_p[11])
        t_1_u = in_p[10]
        t_1_d = 2 * (in_p[0] - in_p[1]) - ((1 - in_p[15])*in_p[4]+in_p[15]*in_p[10])*self.r_h

        return self.n_w*t_0 - t_1_u / t_1_d

    def eqn_h_p2(self, in_p):
        """
        Equation determining the hydration shell parameter h_p2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p+self.f_p2)*(self.m_p-in_p[3]-in_p[4]-in_p[5])
        t_1_u = in_p[5]
        t_1_d = in_p[1]-((1-in_p[15])*in_p[5]+in_p[15]*in_p[11])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_p2(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp+self.f_bp2)*(self.m_bp-in_p[9]-in_p[10]-in_p[11])
        t_1_u = in_p[11]
        t_1_d = in_p[1]-((1-in_p[15])*in_p[5]+in_p[15]*in_p[11])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_h_m0(self, in_p):
        """
        Equation determining the hydration shell parameter h_m0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m)*(self.m_m-in_p[6]-in_p[7]-in_p[8])
        t_1_u = in_p[6]
        t_1_d = 1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_m0(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm) * (self.m_bm - in_p[12] - in_p[13] - in_p[14])
        t_1_u = in_p[12]
        t_1_d = 1-2*in_p[0]+in_p[2]-((1-in_p[15])*in_p[6]+in_p[15]*in_p[12])*self.r_h

        return self.n_w*t_0 - t_1_u / t_1_d

    def eqn_h_m1(self, in_p):
        """
        Equation determining the hydration shell parameter h_m1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m+self.f_m1)*(self.m_m-in_p[6]-in_p[7]-in_p[8])
        t_1_u = in_p[7]
        t_1_d = 2*(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_m1(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm + self.f_bm1)*(self.m_bm-in_p[12]-in_p[13]-in_p[14])
        t_1_u = in_p[13]
        t_1_d = 2 *(in_p[0]-in_p[2])-((1-in_p[15])*in_p[7]+in_p[15]*in_p[13])*self.r_h

        return self.n_w*t_0 - t_1_u / t_1_d

    def eqn_h_m2(self, in_p):
        """
        Equation determining the hydration shell parameter h_m2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m+self.f_m2)*(self.m_m-in_p[6]-in_p[7]-in_p[8])
        t_1_u = in_p[8]
        t_1_d = in_p[2]-((1-in_p[15])*in_p[8]+in_p[15]*in_p[14])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_hb_m2(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm+self.f_bm2)*(self.m_bm-in_p[12]-in_p[13]-in_p[14])
        t_1_u = in_p[14]
        t_1_d = in_p[2]-((1-in_p[15])*in_p[8]+in_p[15]*in_p[14])*self.r_h

        return self.n_w*t_0 -  t_1_u/t_1_d

    def eqn_bjerrum(self, in_p):
        """
        Equation determining the bjerrum equations

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        return self.mu_sf(in_p)-self.mu_bf(in_p)

    def eqns(self, in_p):
        """
         mean field equations

        :param ini_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        eqns=np.array([self.eqn_y(in_p), self.eqn_za(in_p), self.eqn_zd(in_p),
                       self.eqn_h_p0(in_p), self.eqn_h_p1(in_p), self.eqn_h_p2(in_p),
                       self.eqn_h_m0(in_p), self.eqn_h_m1(in_p), self.eqn_h_m2(in_p),
                       self.eqn_hb_p0(in_p), self.eqn_hb_p1(in_p), self.eqn_hb_p2(in_p),
                       self.eqn_hb_m0(in_p), self.eqn_hb_m1(in_p), self.eqn_hb_m2(in_p),
                       self.eqn_bjerrum(in_p)])

        return eqns

    def solve_eqns(self, ini_condition_p, num_eqns=np.arange(16, dtype='int')):
        """
        Solve the mean field equations

        :param ini_condition_p: initial 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        :param num_eqns: indexes of the equations to solve
        """

        ini_c = ini_condition_p[num_eqns]
        def fun(ini_p):
            ini_val = ini_condition_p
            ini_val[num_eqns] = ini_p
            return self.eqns(ini_val)[num_eqns]

        sol = fsolve(fun, ini_c)

        return sol

    def solve_eqns_water_analytical(self):
        """
        Provides the analytical solution for the hydrogen bond quantities for pure water
        """

        df = self.f_w
        y_val = 1+0.25*np.exp(-df)*(1-np.sqrt(1+8*np.exp(df)))
        return np.array([y_val, y_val**2, y_val**2])

    @staticmethod
    def tau_debye(x):
        """
        Function

        :math:`\\tau(x)=\\frac{3}{x^3}\\left(\\log(x)-x+\\frac{x^2}{2}\\right)`

        necessary to compute the debye-huckel contribution
        """

        return 3.0*(np.log(1+x)-x+x**2/2)/x**3

    @staticmethod
    def r_debye(x):
        """
        Function

        :math:`\\frac{3}{x^3}\\left(1+x-\\frac{1}{1+x}-2\\log(1+x)\\right)`

        necessary to compute the water chemical potential
        """

        return 3*(1+x-1/(1+x)-2*np.log(1+x))/x**3
