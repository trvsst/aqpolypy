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

    def __init__(self, nw_i, ns_i, temp, param_w, param_salt, param_h, press=1.01325):

        """
        The constructor, with the following parameters

        :param nw_i: water number density
        :param ns_i: salt number density
        :param temp: temperature in Kelvin
        :param param_w: water parameters (see definition below)
        :param param_salt: salt parameters (see definition below)
        :param param_h: hydration layer parameters, see below
        :param press: pressure in bars, default is 1 atm

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

        # constants
        self.delta_w = un.delta_w()

        # temperature
        self.tp = temp

        # water model at the given temperature and pressure
        self.press = press
        wfm = fm.WaterPropertiesFineMillero(self.tp, self.press)
        self.a_gamma = 3*wfm.a_phi()

        # molecular volume
        self.u_w = param_w['v_w']
        self.u_s = param_salt['v_s']
        self.u_b = param_salt['v_b']

        # concentration in dimensionless units
        self.n_w = nw_i * self.u_w
        self.n_s = ns_i * self.u_w
        self.r_h = self.n_s / self.n_w
        # concentration in the molal scale and ionic strength
        self.ml = self.concentration_molal()
        self.sqrt_i_str = np.sqrt(self.ml)

        # pressure
        self.pvt = self.press*self.u_w/self.tp

        # energies and entropies
        self.e_w = param_w['de_w']
        self.s_w = param_w['se_w']
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
        self.f_bm2 = self.e_bp2/self.tp - self.s_bm2

        # define new hydration parameters
        self.h_p = self.h_p0+self.h_p1+self.h_p2
        self.h_m = self.h_m0 + self.h_m1 + self.h_m2
        self.hb_p = self.hb_p0 + self.hb_p1 + self.hb_p2
        self.hb_m = self.hb_m0 + self.hb_m1 + self.hb_m2

    def f_assoc(self, y, za, zd, fb):
        """
        Defines the association free energy

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        n_w = self.n_w
        n_s = self.n_s

        t_0 = - n_w*(2*y*self.f_w+zd*self.f_2d+za*self.f_2a)-n_s*fb*self.f_bj

        t_1_0 = - n_s*(self.h_m*self.f_m+self.h_p*self.f_p)*(1-fb)
        t_1_1 = - n_s * (self.hb_m * self.f_bm + self.hb_p * self.f_bp) * fb
        t_1 = t_1_0 + t_1_1

        t_2_0 = - n_s*(self.h_m1*self.f_m1+self.h_p1*self.f_p1)*(1-fb)
        t_2_1 = - n_s*(self.h_m2*self.f_m2+self.h_p2*self.f_p2)*(1-fb)
        t_2_2 = - n_s * (self.hb_m1 * self.f_bm1 + self.hb_p1 * self.f_bp1) * fb
        t_2_3 = - n_s * (self.hb_m2 * self.f_bm2 + self.hb_p2 * self.f_bp2) * fb
        t_2 = t_2_0+t_2_1+t_2_2+t_2_3

        va0 = (1-2*y+za) * n_w - ((1-fb) * self.h_p0 + fb * self.hb_p0) * n_s
        t_3 = lg(va0, va0)

        va1 = 2 * (y-za) * n_w - ((1-fb) * self.h_p1 + fb * self.hb_p1) * n_s
        t_4 = lg(va1, va1)-2*(y-za)*n_w*np.log(2)

        va2 = za * n_w - ((1 - fb) * self.h_p2 + fb * self.hb_p2) * n_s
        t_5 = lg(va2, va2)

        vah = lg(self.h_p0, self.h_p0)+lg(self.h_p1, self.h_p1)+lg(self.h_p2, self.h_p2)
        t_6 = (1-fb)*n_s*(vah-lg(self.h_p, self.h_p))

        vabh1 = lg(self.hb_p0, self.hb_p0) + lg(self.hb_p1, self.hb_p1)
        vabh2 = lg(self.hb_p2, self.hb_p2)
        t_7 = fb * n_s * (vabh1 + vabh2 - lg(self.hb_p, self.hb_p))

        vd0 = (1-2*y+zd) * n_w - ((1-fb) * self.h_m0 + fb * self.hb_m0) * n_s
        t_8 = lg(vd0, vd0)

        vd1 = 2 * (y - zd) * n_w - ((1 - fb) * self.h_m1 + fb * self.hb_m1) * n_s
        t_9 = lg(vd1, vd1) - 2 * (y - zd) * n_w * np.log(2)

        vd2 = za * n_w - ((1 - fb) * self.h_m2 + fb * self.hb_m2) * n_s
        t_10 = lg(vd2, vd2)

        vdh = lg(self.h_m0, self.h_m0) + lg(self.h_m1, self.h_m1) + lg(self.h_m2, self.h_m2)
        t_11 = (1 - fb) * n_s * (vdh - lg(self.h_m, self.h_m))

        vdbh1 = lg(self.hb_m0, self.hb_m0) + lg(self.hb_m1, self.hb_m1)
        vdbh2 = lg(self.hb_m2, self.hb_m2)
        t_12 = fb * n_s * (vdbh1 + vdbh2 - lg(self.hb_m, self.hb_m))

        t_13_1 = n_s*self.m_p*(1-fb)*lg(1-self.h_p/self.m_p, 1-self.h_p/self.m_p)
        t_13_2 = n_s*self.m_p*(1-fb)*lg(self.h_p/self.m_p, self.h_p/self.m_p)
        t_13_3 = n_s*self.m_bp*fb*lg(1 - self.hb_p / self.m_bp, 1 - self.hb_p / self.m_bp)
        t_13_4 = n_s*self.m_bp*fb*lg(self.hb_p / self.m_bp, self.hb_p / self.m_bp)
        t_13 = t_13_1+t_13_2+t_13_3+t_13_4

        t_14_1 = n_s*self.m_m*(1-fb)*lg(1-self.h_m/self.m_m, 1-self.h_m/self.m_m)
        t_14_2 = n_s*self.m_m*(1-fb)*lg(self.h_m/self.m_m, self.h_m/self.m_m)
        t_14_3 = n_s*self.m_bm*fb*lg(1 - self.hb_m / self.m_bm, 1 - self.hb_m / self.m_bm)
        t_14_4 = n_s*self.m_bp*fb*lg(self.hb_m / self.m_bm, self.hb_m / self.m_bm)
        t_14 = t_14_1 + t_14_2 + t_14_3 + t_14_4

        t_15_1 = n_s*fb*np.log(self.m_bm*self.m_bp)-2*lg(y*n_w, y*n_w)
        t_15_2 = 2*n_s*(lg(1-fb, 1-fb)+lg(fb, fb))-lg(n_s*fb, n_s*fb)
        t_15 = t_15_1+t_15_2

        t_16 = -2*lg(n_w, n_w)-2*y*n_w*(np.log(2)-1)+n_s*(fb * (1 + self.hb_p + self.hb_m) + (1 - fb) * (self.h_p + self.h_m))

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

    def f_comp(self, fb, k_ref):
        """
        Defines the compressible free energy

        :param fb: fraction of Bjerrum pairs
        :param k_ref: reference compressibility
        """

        k_param = self.u_w/(self.tp*k_ref)
        v0 = self.n_w+((1-fb)*self.u_s+fb*self.u_b)*self.n_s/self.u_w

        return 0.5*k_param*(1-v0)**2/v0

    def f_debye(self, fb, b_g=1e-2):
        """
        Defines the Debye-Huckel contribution

        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the free energy
        """

        i_str = np.sqrt(1-fb)*self.sqrt_i_str
        val = b_g*i_str

        return -4*self.a_gamma*i_str**3*self.tau_debye(val)*self.n_w/(3*self.delta_w)

    def f_total(self, y, za, zd, fb, k_ref, b_g=1e-4):
        """
        Defines the total free energy

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param k_ref: reference compressibility for the compressibility free energy
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        f_i = self.f_ideal()
        f_a = self.f_assoc(y, za, zd, fb)
        f_c = self.f_comp(fb, k_ref)
        f_d = self.f_debye(fb, b_g)

        return f_i + f_a + f_c + f_d

    def mu_w_1(self, y, za, zd, fb):
        """
        Defines partial contribution to the water chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        r_h = self.r_h
        n_w = self.n_w
        n_s = self.n_s

        t_11 = (1-2*y)*np.log(n_w)
        t_12 = (1-2*y+za)*np.log(1 - 2 * y + za - ((1-fb) * self.h_p0 + fb * self.hb_p0) * r_h)
        t_1 = t_11+t_12

        t_2 = 2*(y-za)*np.log(2 * (y-za) - ((1-fb) * self.h_p1 + fb * self.hb_p1) * r_h)

        t_3 = za*np.log(za - ((1-fb) * self.h_p2 + fb * self.hb_p2) * r_h)

        t_4 = (1-2*y+zd)*np.log(1 - 2 * y + zd - ((1-fb) * self.h_m0 + fb * self.hb_m0) * r_h)

        t_5 = 2*(y-zd)*np.log(2 * (y-zd) - ((1-fb) * self.h_m1 + fb * self.hb_m1) * r_h)

        t_6 = zd*np.log(zd - ((1-fb) * self.h_m2 + fb * self.hb_m2) * r_h)

        t_7 = -2*y*self.f_w-zd*self.f_2d-za*self.f_2a-2*(3*y-za-zd)*np.log(2)-2*lg(y, y)

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7

    def mu_w_debye(self, fb, b_g):
        """
        Defines the Electrostatic chemical potential

        :param b_g: chemical potential constant
        :param fb: fraction of Bjerrum pairs
        """

        x_val = np.sqrt((1-fb))*self.sqrt_i_str
        return 2*self.a_gamma*x_val**3*self.r_debye(b_g*x_val)/(3*self.delta_w)

    def mu_w_comp(self, y, fb):
        """
        Defines the compressibility chemical potential

        :param y: fraction of water hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """
        n_w = self.n_w
        n_s = self.n_s

        t_1 = self.pvt-(1-2*y)*n_w+((1-fb)*(self.h_p+self.h_m)+fb*(self.hb_p+self.hb_m))*n_s-(2-fb)*n_s

        return t_1

    def mu_sf_1(self, y, za, zd, fb):
        """
        Defines partial contribution to the free salt chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        r_h = self.r_h
        n_w = self.n_w
        n_s = self.n_s

        t_11 = 2*np.log((1-fb)*n_s)-self.h_m*self.f_m-self.h_p*self.f_p
        t_12 = self.h_m1*self.f_m1+self.h_m2*self.f_m2+self.h_p1*self.f_p1+self.h_p2*self.f_p2
        t_1 = t_11 + t_12

        t_2 = -self.h_p0*np.log(1 - 2 * y + za - ((1-fb) * self.h_p0 + fb * self.hb_p0) * r_h)

        t_3 = -self.h_p1*np.log(2 * (y-za) - ((1-fb) * self.h_p1 + fb * self.hb_p1) * r_h)

        t_4 = -self.h_p2*np.log(za - ((1-fb) * self.h_p2 + fb * self.hb_p2) * r_h)

        t_5 = lg(self.h_p0, self.h_p0)+lg(self.h_p1, self.h_p1)+lg(self.h_p2, self.h_p2)-lg(self.h_p, self.h_p)

        t_6 = -self.h_m0*np.log(1 - 2 * y + zd - ((1-fb) * self.h_m0 + fb * self.hb_m0) * r_h)

        t_7 = -self.h_m1*np.log(2 * (y-zd) - ((1-fb) * self.h_m1 + fb * self.hb_m1) * r_h)

        t_8 = -self.h_m2*np.log(zd - ((1-fb) * self.h_m2 + fb * self.hb_m2) * r_h)

        t_9 = lg(self.h_m0, self.h_m0) + lg(self.h_m1, self.h_m1) + lg(self.h_m2, self.h_m2) - lg(self.h_m, self.h_m)

        t_10_1 = self.m_p*(lg(1-self.h_p/self.m_p,1-self.h_p/self.m_p)+ lg(self.h_p/self.m_p,self.h_p/self.m_p))
        t_10_2 = self.m_p*(lg(1-self.h_m/self.m_m,1-self.h_m/self.m_m)+ lg(self.h_m/self.m_m,self.h_m/self.m_m))
        t_10 = t_10_1 + t_10_2

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10

    def mu_sf_debye(self, fb, b_g):
        """
        Defines the salt Electrostatic chemical potential

        :param i_str: ionic strength (molal scale)
        :param b_g: chemical potential constant
        :param fb: fraction of Bjerrum pairs
        """

        x_val = np.sqrt(1-fb)*self.sqrt_i_str

        return -2*self.a_gamma*x_val/(1+b_g*x_val)

    def mu_sb_1(self, y, za, zd, fb):
        """
        Defines partial contribution to the bjerrum salt chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        r_h = self.r_h
        n_w = self.n_w
        n_s = self.n_s

        t_11 = np.log(fb*n_s) - self.hb_m * self.f_bm - self.hb_p * self.f_bp - self.f_bj
        t_12 = self.hb_m1 * self.f_bm1 + self.hb_m2 * self.f_bm2 + self.hb_p1 * self.f_bp1 + self.hb_p2 * self.f_bp2
        t_1 = t_11 + t_12

        t_2 = -self.hb_p0 * np.log(1 - 2 * y + za - ((1 - fb) * self.h_p0 + fb * self.hb_p0) * r_h)

        t_3 = -self.hb_p1 * np.log(2 * (y - za) - ((1 - fb) * self.h_p1 + fb * self.hb_p1) * r_h)

        t_4 = -self.hb_p2 * np.log(za - ((1 - fb) * self.h_p2 + fb * self.hb_p2) * r_h)

        t_5 = lg(self.hb_p0, self.hb_p0) + lg(self.hb_p1, self.hb_p1) + lg(self.hb_p2, self.hb_p2) - lg(self.hb_p, self.hb_p)

        t_6 = -self.hb_m0 * np.log(1 - 2 * y + zd - ((1 - fb) * self.h_m0 + fb * self.hb_m0) * r_h)

        t_7 = -self.hb_m1 * np.log(2 * (y - zd) - ((1 - fb) * self.h_m1 + fb * self.hb_m1) * r_h)

        t_8 = -self.hb_m2 * np.log(zd - ((1 - fb) * self.h_m2 + fb * self.hb_m2) * r_h)

        t_9_1 = lg(self.hb_m0, self.hb_m0) + lg(self.hb_m1, self.hb_m1)
        t_9_2 = lg(self.hb_m2, self.hb_m2) - lg(self.hb_m, self.hb_m)
        t_9 = t_9_1 + t_9_2

        t_10_1 = self.m_bp*(lg(1 - self.hb_p / self.m_bp, 1 - self.hb_p / self.m_bp) + lg(self.hb_p / self.m_bp, self.hb_p / self.m_bp))
        t_10_2 = self.m_bp*(lg(1 - self.hb_m / self.m_bm, 1 - self.hb_m / self.m_bm) + lg(self.hb_m / self.m_bm, self.hb_m / self.m_bm))
        t_10 = t_10_1 + t_10_2

        t_11 = np.log(self.m_m*self.m_p)

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10 + t_11

    def mu_w(self, y, za, zd, fb, b_g=1e-4):
        """
        Defines the water chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        m_1 = self.mu_w_1(y, za, zd, fb)
        m_2 = self.mu_w_debye(fb, b_g)
        m_3 = self.mu_w_comp(y, fb)

        m_total = m_1+m_2+m_3

        return m_total

    def mu_sf(self, y, za, zd, fb, b_g=1e-4):
        """
        Defines the free salt chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        m_1 = self.mu_sf_1(y, za, zd, fb)
        m_2 = self.mu_sf_debye(fb, b_g)
        m_3 = self.mu_w_comp(y, fb)*self.u_w/self.u_s

        m_total = m_1 + m_2 + m_3

        return m_total

    def mu_bf(self, y, za, zd, fb, b_g=1e-4):
        """
        Defines the bjerrum salt chemical potential

        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        m_1 = self.mu_sb_1(y, za, zd, fb)
        m_2 = self.mu_w_comp(y, fb) * self.u_w / self.u_s

        m_total = m_1 + m_2

        return m_total

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
        t_1_u = 1-2*in_p['y']+in_p['zd']-((1-in_p['fb'])*in_p['h_m0']+in_p['fb']*in_p['hb_m0'])*self.r_h
        t_2_u = in_p['zd']-((1-in_p['fb'])*in_p['h_m2']+in_p['fb']*in_p['hb_m2'])*self.r_h
        t_1_d = 2*(in_p['y']-in_p['zd'])-((1-in_p['fb'])*in_p['h_m1']+in_p['fb']*in_p['hb_m1'])*self.r_h

        return t_0-t_1_u*t_2_u/t_1_d**2

    def eqn_h_p0(self, in_p):
        """
        Equation determining the hydration shell parameter h_p0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p)*(self.m_p-in_p['h_p0']-in_p['h_p1']-in_p['h_p2'])
        t_1_u = in_p['h_p0']
        t_1_d = 1-2*in_p['y']+in_p['za']-((1-in_p['fb'])*in_p['h_p0']+in_p['fb']*in_p['hb_p0'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_p0(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp) * (self.m_bp - in_p['hb_p0'] - in_p['hb_p1'] - in_p['hb_p2'])
        t_1_u = in_p['h_p0']
        t_1_d = 1-2*in_p['y']+in_p['za']-((1-in_p['fb'])*in_p['hb_p0']+in_p['fb']*in_p['hb_p0'])*self.r_h

        return t_0 - t_1_u / t_1_d

    def eqn_h_p1(self, in_p):
        """
        Equation determining the hydration shell parameter h_p1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p+self.f_p1)*(self.m_p-in_p['h_p0']-in_p['h_p1']-in_p['h_p2'])
        t_1_u = in_p['h_p1']
        t_1_d = 2*(in_p['y']-in_p['za'])-((1-in_p['fb'])*in_p['h_p1']+in_p['fb']*in_p['hb_p1'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_p1(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp + self.f_bp1) * (self.m_bp - in_p['hb_p0'] - in_p['hb_p1'] - in_p['hb_p2'])
        t_1_u = in_p['hb_p1']
        t_1_d = 2 * (in_p['y'] - in_p['za']) - ((1 - in_p['fb']) * in_p['h_p1'] + in_p['fb'] * in_p['hb_p1'])*self.r_h

        return t_0 - t_1_u / t_1_d

    def eqn_h_p2(self, in_p):
        """
        Equation determining the hydration shell parameter h_p2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_p+self.f_p2)*(self.m_p-in_p['h_p0']-in_p['h_p1']-in_p['h_p2'])
        t_1_u = in_p['h_p2']
        t_1_d = in_p['za']-((1-in_p['fb'])*in_p['h_p1']+in_p['fb']*in_p['hb_p1'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_p2(self, in_p):
        """
        Equation determining the hydration shell parameter hb_p2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bp+self.f_bp2)*(self.m_p-in_p['hb_p0']-in_p['hb_p1']-in_p['hb_p2'])
        t_1_u = in_p['h_p2']
        t_1_d = in_p['za']-((1-in_p['fb'])*in_p['h_p2']+in_p['fb']*in_p['hb_p2'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_h_m0(self, in_p):
        """
        Equation determining the hydration shell parameter h_m0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m)*(self.m_m-in_p['h_m0']-in_p['h_m1']-in_p['h_m2'])
        t_1_u = in_p['h_m0']
        t_1_d = 1-2*in_p['y']+in_p['zd']-((1-in_p['fb'])*in_p['h_m0']+in_p['fb']*in_p['hb_m0'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_m0(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m0

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm) * (self.m_bm - in_p['hb_m0'] - in_p['hb_m1'] - in_p['hb_m2'])
        t_1_u = in_p['h_p0']
        t_1_d = 1-2*in_p['y']+in_p['zd']-((1-in_p['fb'])*in_p['hb_m0']+in_p['fb']*in_p['hb_m0'])*self.r_h

        return t_0 - t_1_u / t_1_d

    def eqn_h_m1(self, in_p):
        """
        Equation determining the hydration shell parameter h_m1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m+self.f_m1)*(self.m_m-in_p['h_m0']-in_p['h_m1']-in_p['h_m2'])
        t_1_u = in_p['h_m1']
        t_1_d = 2*(in_p['y']-in_p['zd'])-((1-in_p['fb'])*in_p['h_m1']+in_p['fb']*in_p['hb_m1'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_m1(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m1

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm + self.f_bm1) * (self.m_bm - in_p['hb_m0'] - in_p['hb_m1'] - in_p['hb_m2'])
        t_1_u = in_p['hb_m1']
        t_1_d = 2 *(in_p['y']-in_p['zd'])-((1 - in_p['fb'])*in_p['h_m1']+in_p['fb']*in_p['hb_m1'])*self.r_h

        return t_0 - t_1_u / t_1_d

    def eqn_h_m2(self, in_p):
        """
        Equation determining the hydration shell parameter h_m2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_m+self.f_m2)*(self.m_m-in_p['h_m0']-in_p['h_m1']-in_p['h_m2'])
        t_1_u = in_p['h_m2']
        t_1_d = in_p['zd']-((1-in_p['fb'])*in_p['h_p1']+in_p['fb']*in_p['hb_p1'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_hb_m2(self, in_p):
        """
        Equation determining the hydration shell parameter hb_m2

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        t_0 = np.exp(self.f_bm+self.f_bm2)*(self.m_m-in_p['hb_m0']-in_p['hb_m1']-in_p['hb_m2'])
        t_1_u = in_p['h_m2']
        t_1_d = in_p['zd']-((1-in_p['fb'])*in_p['h_m2']+in_p['fb']*in_p['hb_m2'])*self.r_h

        return t_0 -  t_1_u/t_1_d

    def eqn_bjerrum(self, in_p):
        """
        Equation determining the bjerrum equations

        :param in_p: 16 parameters, [y,za,zd,h+..h-..hb+..hb-,fb]
        """

        return 1.0

    def concentration_molal(self):
        """
        returns the concentration in molal units

        :param nw_i: water number density
        :param ns_i: salt number density
        """
        return self.delta_w * self.r_h

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
