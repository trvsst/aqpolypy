"""
:module: BrushSolution
:platform: Unix, Windows, OS
:synopsis: Uses Gibbs Duhem relation to compute activity coefficient

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, August2023
.. history:
..
"""

import numpy as np
import aqpolypy.water.WaterMilleroBP as fm

class GibbsDuhem(object):

    """ Defines Gibbs Duhem relation"""

    def __init__(self, m_l, osmotic_coeff, log_gamma, temp):
        """
        constructor

        :param m_l: molality
        :param osmotic_coeff: osmotic coefficient
        :param log_gamma: activity coefficient
        :param temp: temperature in kelvin
        """

        # water model at the given temperature and pressure
        self.press = 1.01325
        wfm = fm.WaterPropertiesFineMillero(temp, self.press)
        self.a_gamma = 3*wfm.a_phi()

        self.m_l = m_l
        self.osmotic_coeff = osmotic_coeff
        self.log_gamma = log_gamma

    def calculate_log_gamma(self):
        """
        calculates the activity coefficient from the give osmotic coefficient
        """

        return self._intg_osm()

    def calculate_osmotic_coeff(self):
        """
        calculates the osmotic coefficient
        """

        return self._intg_lg()

    def _intg_osm(self):
        """
        Helper function to compute the integral present
        """

        delta_osm = np.zeros_like(self.m_l)
        c_activity = np.zeros_like(self.m_l)

        delta_osm[:] = self.osmotic_coeff[:]-1+self.a_gamma*np.sqrt(self.m_l[:])/3.0
        c_0 = delta_osm[0]
        for ind, ml in enumerate(self.m_l):
            delta_p = np.trapz(delta_osm[:ind]/self.m_l[:ind], self.m_l[:ind])
            c_activity[ind] = -self.a_gamma*np.sqrt(ml)+delta_osm[ind]+delta_p

        return c_activity+c_0

    def _intg_lg(self):
        """
        Helper function to compute the integral present
        """

        delta_gamma = np.zeros_like(self.m_l)
        c_osmotic = np.zeros_like(self.m_l)

        delta_gamma[:] = self.log_gamma[:]+self.a_gamma*np.sqrt(self.m_l[:])
        c_0=-delta_gamma[0]
        for ind, ml in enumerate(self.m_l):
            delta_p = np.trapz(delta_gamma[:ind], self.m_l[:ind])
            c_osmotic[ind] =  1-self.a_gamma*np.sqrt(ml)/3.0+delta_gamma[ind]-delta_p/ml

        return c_osmotic+c_0

    def max_diff_gamma(self):
        """
        returns the molality and maximum difference between calculated and given activity coefficients
        """

        ind = np.argmax(np.abs(self.calculate_log_gamma()-self.log_gamma))
        val_max = np.max(np.abs(self.calculate_log_gamma()-self.log_gamma))
        val = (self.m_l[ind], val_max, ind)

        return val

