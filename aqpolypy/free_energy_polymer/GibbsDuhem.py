"""
:module: BrushSolution
:platform: Unix, Windows, OS
:synopsis: Uses Gibbs Duhem relation to compute activity coefficient

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, August2023
.. history:
..
"""

import numpy as np


class GibbsDuhem(object):

    """ Defines Gibbs Duhem relation"""

    def __init__(self, m_l, osmotic_coeff, log_gamma, min_molality=1e-3):
        """
        constructor

        :param m_l: molality
        :param osmotic_coeff: osmotic coefficient
        :param log_gamma: activity coefficient
        :param min_molality: check for molalities larger than min_molaity only
        """

        self.m_l = m_l
        self.osmotic_coeff = osmotic_coeff
        self.log_gamma = log_gamma
        self.ind_min = np.argmin((self.m_l-min_molality)**2)

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

        c_int = np.zeros_like(self.m_l)
        c_res = np.zeros_like(self.m_l)

        if self.ind_min == 0:
            osm_0 = 1
            gam_0 = 0.0
        else:
            osm_0 = self.osmotic_coeff[self.ind_min]
            gam_0 = self.log_gamma[self.ind_min]

        for ind in range(self.ind_min, c_int.shape[0]):
            m_val = self.m_l[self.ind_min:(ind+1)]
            val1 = np.trapz((self.osmotic_coeff[:(ind+1)]-1)/self.m_l[:(ind+1)],self.m_l[:(ind+1)])
            val2 = np.trapz((self.osmotic_coeff[:(self.ind_min+1)]-1)/self.m_l[:(self.ind_min+1)],self.m_l[:(self.ind_min+1)])
            c_int[ind] =  val1 - val2

            c_res = gam_0 + self.osmotic_coeff-osm_0+c_int
            c_res[:self.ind_min] = self.log_gamma[:self.ind_min]

            return c_res
    def _intg_lg(self):
        """
        Helper function to compute the integral present
        """

        c_val = np.zeros_like(self.m_l)
        c_act = np.zeros(self.m_l.shape[0]+1)
        c_int = np.zeros_like(c_act)
        c_act[1:] = self.m_l[:]
        c_int[1:] = self.log_gamma[:]

        for ind in range(1, c_int.shape[0]):
            c_val[ind-1] =  1 + np.trapz(c_act[:(ind+1)], c_int[:(ind+1)])/self.m_l[ind-1]

        return c_val

    def max_diff_gamma(self):
        """
        returns the molality and maximum difference between calculated and given activity coefficients
        """

        ind = np.argmax(np.abs(self.calculate_log_gamma()-self.log_gamma))
        val_max = np.max(np.abs(self.calculate_log_gamma()-self.log_gamma))
        val = (self.m_l[ind], val_max, ind)

        return val

