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
    Class defining an electrolyte solution
    """

    def __init__(self, temp, param_w, param_salt, param_h, press=1.01325):

        """
        The constructor, with the following parameters

        :param temp: temperature in Kelvin
        :param param_w: water parameters (see definition below)
        :param param_salt: salt parameters (see definition below)
        :param param_h: hydration layer parameters, see below
        :param press: pressure in bars, default is 1 atm

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

        # water model at the given temperature
        self.press = press
        wfm = fm.WaterPropertiesFineMillero(self.tp, self.press)
        self.a_gamma = 3*wfm.a_phi()

        # molecular volume
        self.u_w = param_w['v_w']
        self.u_s = param_salt['v_s']
        self.u_b = param_salt['v_b']

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
        self.h_bp0 = param_h['hb_p0']
        self.h_bp1 = param_h['hb_p1']
        self.h_bp2 = param_h['hb_p2']
        self.h_bm0 = param_h['hb_m0']
        self.h_bm1 = param_h['hb_m1']
        self.h_bm2 = param_h['hb_m2']

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
        self.f_bp = self.e_bp / self.tp - self.s_bp
        self.f_bp1 = self.e_bp1 / self.tp - self.s_bp1
        self.f_bp2 = self.e_bp2 / self.tp - self.s_bp2
        self.f_bm = self.e_bm / self.tp - self.s_bm
        self.f_bm1 = self.e_bm1 / self.tp - self.s_bm1
        self.f_bm2 = self.e_bp2 / self.tp - self.s_bm2

        # define new hydration parameters
        self.h_p = self.h_p0+self.h_p1+self.h_p2
        self.h_m = self.h_m0 + self.h_m1 + self.h_m2
        self.h_bp = self.h_bp0 + self.h_bp1 + self.h_bp2
        self.h_bm = self.h_bm0 + self.h_bm1 + self.h_bm2

    def f_assoc(self, nw_i, ns_i, y, za, zd, fb):
        """
        Defines the association free energy

        :param nw_i: water number density
        :param ns_i: salt number density
        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        n_w = nw_i * self.u_w
        n_s = ns_i * self.u_w

        t_0 = - n_w*(2*y*self.f_w+zd*self.f_2d+za*self.f_2a)-n_s*fb*self.f_bj

        t_1_0 = - n_s*(self.h_m*self.f_m+self.h_p*self.f_p)*(1-fb)
        t_1_1 = - n_s*(self.h_bm*self.f_bm+self.h_bp*self.f_bp)*fb
        t_1 = t_1_0 + t_1_1

        t_2_0 = - n_s*(self.h_m1*self.f_m1+self.h_p1*self.f_p1)*(1-fb)
        t_2_1 = - n_s*(self.h_m2*self.f_m2+self.h_p2*self.f_p2)*(1-fb)
        t_2_2 = - n_s*(self.h_bm1*self.f_bm1+self.h_bp1*self.f_bp1)*fb
        t_2_3 = - n_s*(self.h_bm2 * self.f_bm2 + self.h_bp2 * self.f_bp2)*fb
        t_2 = t_2_0+t_2_1+t_2_2+t_2_3

        va0 = (1-2*y+za)*n_w-((1-fb)*self.h_p0+fb*self.h_bp0)*n_s
        t_3 = lg(va0, va0)

        va1 = 2*(y-za)*n_w-((1-fb)*self.h_p1+fb*self.h_bp1)*n_s
        t_4 = lg(va1, va1)-2*(y-za)*n_w*np.log(2)

        va2 = za * n_w - ((1 - fb) * self.h_p2 + fb * self.h_bp2)*n_s
        t_5 = lg(va2, va2)

        vah = lg(self.h_p0, self.h_p0)+lg(self.h_p1, self.h_p1)+lg(self.h_p2, self.h_p2)
        t_6 = (1-fb)*n_s*(vah-lg(self.h_p, self.h_p))

        vabh1 = lg(self.h_bp0, self.h_bp0) + lg(self.h_bp1, self.h_bp1)
        vabh2 = lg(self.h_bp2, self.h_bp2)
        t_7 = fb * n_s * (vabh1+vabh2-lg(self.h_bp, self.h_bp))

        vd0 = (1-2*y+zd)*n_w-((1-fb)*self.h_m0+fb*self.h_bm0)*n_s
        t_8 = lg(vd0, vd0)

        vd1 = 2 * (y - zd) * n_w - ((1 - fb) * self.h_m1 + fb * self.h_bm1) * n_s
        t_9 = lg(vd1, vd1) - 2 * (y - zd) * n_w * np.log(2)

        vd2 = za * n_w - ((1 - fb) * self.h_m2 + fb * self.h_bm2) * n_s
        t_10 = lg(vd2, vd2)

        vdh = lg(self.h_m0, self.h_m0) + lg(self.h_m1, self.h_m1) + lg(self.h_m2, self.h_m2)
        t_11 = (1 - fb) * n_s * (vdh - lg(self.h_m, self.h_m))

        vdbh1 = lg(self.h_bm0, self.h_bm0) + lg(self.h_bm1, self.h_bm1)
        vdbh2 = lg(self.h_bm2, self.h_bm2)
        t_12 = fb * n_s * (vdbh1 + vdbh2 - lg(self.h_bm, self.h_bm))

        t_13_1 = n_s*self.m_p*(1-fb)*lg(1-self.h_p/self.m_p, 1-self.h_p/self.m_p)
        t_13_2 = n_s*self.m_p*(1-fb)*lg(self.h_p/self.m_p, self.h_p/self.m_p)
        t_13_3 = n_s*self.m_bp*fb*lg(1-self.h_bp/self.m_bp, 1-self.h_bp/self.m_bp)
        t_13_4 = n_s*self.m_bp*fb*lg(self.h_bp/self.m_bp, self.h_bp/self.m_bp)
        t_13 = t_13_1+t_13_2+t_13_3+t_13_4

        t_14_1 = n_s*self.m_m*(1-fb)*lg(1-self.h_m/self.m_m, 1-self.h_m/self.m_m)
        t_14_2 = n_s*self.m_m*(1-fb)*lg(self.h_m/self.m_m, self.h_m/self.m_m)
        t_14_3 = n_s*self.m_bm*fb*lg(1-self.h_bm/self.m_bm, 1-self.h_bm/self.m_bm)
        t_14_4 = n_s*self.m_bp*fb*lg(self.h_bm/self.m_bm, self.h_bm/self.m_bm)
        t_14 = t_14_1 + t_14_2 + t_14_3 + t_14_4

        t_15_1 = n_s*fb*np.log(self.m_bm*self.m_bp)-2*lg(y*n_w, y*n_w)
        t_15_2 = 2*n_s*(lg(1-fb, 1-fb)+lg(fb, fb))-lg(n_s*fb, n_s*fb)
        t_15 = t_15_1+t_15_2

        t_16 = -2*lg(n_w, n_w)-2*y*n_w*(np.log(2)-1)+n_s*(fb*(1+self.h_bp+self.h_bm)+(1-fb)*(self.h_p+self.h_m))

        t_s = t_0+t_1+t_2+t_3+t_4+t_5+t_6+t_7+t_8+t_9+t_10+t_11+t_12+t_13+t_14+t_15+t_16

        return t_s

    def f_ideal(self, nw_i, ns_i):
        """
        Defines the ideal free energy

        :param nw_i: water number density
        :param ns_i: salt number density
        """

        n_w = nw_i * self.u_w
        n_s = ns_i * self.u_w

        t_s = lg(n_w, n_w) + 2*lg(n_s, n_s) - (n_w+2*n_s)

        return t_s

    def f_comp(self, nw_i, ns_i, fb, k_ref):
        """
        Defines the compressible free energy

        :param nw_i: water number density
        :param ns_i: salt number density
        :param fb: fraction of Bjerrum pairs
        :param k_ref: reference compressibility
        """

        k_param = self.u_w/(self.tp*k_ref)
        v0 = nw_i * self.u_w+((1-fb)*self.u_s+fb*self.u_b)*ns_i

        return 0.5*k_param*(1-v0)**2/v0

    def f_debye(self, nw_i, ns_i, fb, b_g=1e-2):
        """
        Defines the Debye-Huckel contribution

        :param nw_i: water number density
        :param ns_i: salt number density
        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the free energy
        """

        molal = self.concentration_molal(nw_i, ns_i)

        i_str = np.sqrt((1-fb)*molal)
        val = b_g*i_str

        return -4*self.a_gamma*i_str**3*self.tau_debye(val)*nw_i/(3*self.delta_w)

    def f_total(self, nw_i, ns_i, y, za, zd, fb, k_ref, b_g=1e-4):
        """
        Defines the total free energy

        :param nw_i: water number density
        :param ns_i: salt number density
        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param k_ref: reference compressibility for the compressibility free energy
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        f_i = self.f_ideal(nw_i, ns_i)
        f_a = self.f_assoc(nw_i, ns_i, y, za, zd, fb)
        f_c = self.f_comp(nw_i, ns_i, fb, k_ref)
        f_d = self.f_debye(nw_i, ns_i, fb, b_g)

        return f_i + f_a + f_c + f_d

    def mu_w_1(self, nw_i, ns_i, y, za, zd, fb):
        """
        Defines partial contribution to the water chemical potential

        :param nw_i: water number density
        :param ns_i: salt number density
        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """

        r_h = ns_i/nw_i
        n_w = nw_i * self.u_w
        n_s = ns_i * self.u_w

        t_11 = (1-2*y)*np.log(n_w)
        t_12 = (1-2*y+za)*np.log(1-2*y+za-((1-fb)*self.h_p0+fb*self.h_bp0)*r_h)
        t_1 = t_11+t_12

        t_2 = 2*(y-za)*np.log(2*(y-za)-((1-fb)*self.h_p1+fb*self.h_bp1)*r_h)

        t_3 = za*np.log(za-((1-fb)*self.h_p2+fb*self.h_bp2)*r_h)

        t_4 = (1-2*y+zd)*np.log(1-2*y+zd-((1-fb)*self.h_m0+fb*self.h_bm0)*r_h)

        t_5 = 2*(y-zd)*np.log(2*(y-zd)-((1-fb)*self.h_m1+fb*self.h_bm1)*r_h)

        t_6 = zd*np.log(zd-((1-fb)*self.h_m2+fb*self.h_bm2)*r_h)

        t_7 = -2*y*self.f_w-zd*self.f_2d-za*self.f_2a-2*(3*y-za-zd)*np.log(2)-2*lg(y, y)

        return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7

    def mu_w_debye(self, i_str, fb, b_g):
        """
        Defines the Electrostatic chemical potential

        :param i_str: ionic strength (molal scale)
        :param b_g: chemical potential constant
        :param fb: fraction of Bjerrum pairs
        """

        x_val = np.sqrt((1-fb)*i_str)
        return 2*self.a_gamma*x_val**3*self.r_debye(b_g*x_val)/(3*self.delta_w)

    def mu_w_comp(self, nw_i, ns_i, y, fb):
        """
        Defines the compressibility chemical potential

        :param nw_i: water number density
        :param ns_i: salt number density
        :param y: fraction of water hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        """
        n_w = nw_i * self.u_w
        n_s = ns_i * self.u_w

        t_1 = -(1-2*y)*n_w +((1-fb)*(self.h_p+self.h_m)+fb*(self.h_bp+self.h_bm))*n_s-(2-fb)*n_s

        return t_1

    def mu_w(self, nw_i, ns_i, y, za, zd, fb, b_g=1e-4):
        """
        Defines the water chemical potential

        :param nw_i: water number density
        :param ns_i: salt number density
        :param y: fraction of water hydrogen bonds
        :param za: fraction of double acceptor hydrogen bonds
        :param zd: fraction of double donor hydrogen bonds
        :param fb: fraction of Bjerrum pairs
        :param b_g: parameter defining the extension for the electrostatic free energy
        """

        m_1 = self.mu_w_1(nw_i, ns_i, y, za, zd, fb)

        i_str = self.concentration_molal(nw_i, ns_i)
        m_2 = self.mu_w_debye(i_str, fb, b_g)

        m_3 = self.mu_w_comp(nw_i, ns_i, y, fb)

        m_total = m_1+m_2+m_3

        return m_total

    def concentration_molal(self, nw_i, ns_i):
        """
        returns the concentration in molal units

        :param nw_i: water number density
        :param ns_i: salt number density
        """
        return self.delta_w * ns_i / nw_i

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
