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

import aqpolypy.units.concentration as con
import aqpolypy.water.WaterMilleroBP as wbp


class ElectrolyteSolution(object):
    """
    Class defining an electrolyte solution
    """

    def __init__(self, param_w, param_salt):

        """
        The constructor, with the following parameters

        :param param_w: water parameters (see definition below)
        :param param_salt: salt parameters (see definition below)

        the parameters param_w is a dictionary with
        :math:`v_w = \\upsilon_w, de_w = \\Delta E_w, ds_w = \\Delta S_w, de_2d=\\Delta E_{2d}, \
        ds_2d = \\Delta S_{2d}, de_2a=\\Delta E_{2a}, ds_2a = \\Delta S_{2a}`

        the parameters param_salt is a dictionary given by
        :math:`de_p0 = \\Delta E_{(+,0)}, ds_p0=\\Delta S_{(+,0)}, de_p1 = \\Delta E_{(+,1)}, ds_p1=\\Delta S_{(+,1)} \
        ,de_p2 = \\Delta E_{(+,2)}, ds_p2=\\Delta S_{(+,2)}, de_m0=\\Delta E_{(-,0)}, ds_m0=\\Delta S_{(-,0)}, \
        de_m1 = \\Delta E_{(-,1)}, ds_m1=\\Delta S_{(-,1)}, de_m2 = \\Delta E_{(-,2)}, ds_m2=\\Delta S_{(-,2)} \
        m_p = M_{+}, m_- = M_{-}, mb_p = M_{+}^B, mb_m = M_{-}^B`
        """

        # molecular volume
        self.u_w = param_w['v_w']

        # energies
        self.e_w = param_w['de_w']
        self.s_w = param_w['se_w']
        self.e_2d = param_w['de_2w']
        self.s_2d = param_w['ds_2w']
        self.e_2a = param_w['de_2a']
        self.s_2a = param_w['ds_2a']

        self.e_p = param_salt['de_p0']
        self.s_p = param_salt['ds_p0']
        self.e_p1 = param_salt['de_p1']
        self.s_p1 = param_salt['ds_p1']
        self.e_p2 = param_salt['de_p2']
        self.s_p2 = param_salt['ds_p2']
        self.e_m = param_salt['de_m0']
        self.s_m = param_salt['ds_m0']
        self.e_m1 = param_salt['de_m1']
        self.s_m1 = param_salt['ds_m1']
        self.e_m2 = param_salt['de_m2']
        self.s_m2 = param_salt['ds_m2']
        self.m_p = param_salt['m_p']
        self.m_m = param_salt['m_m']
        self.mb_p = param_salt['mb_p']
        self.mb_m = param_salt['mb_m']



