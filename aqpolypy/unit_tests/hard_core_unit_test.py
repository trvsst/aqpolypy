"""
:module: hard_core_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for HardCore

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""
import numpy as np
import unittest
import aqpolypy.salts_theory.HardCore as hc


class TestHardCore(unittest.TestCase):

    def test_free_energy_hc_excess(self):
        # parameters in [[concentration of the different species (M)], ion size (A), free energy excess]
        param = np.array([[np.array([2.4, 2.4, 0.6]), 4, -0.00085523],
                          [np.array([2.4, 2.4, 0.6]), 6, -0.00493081],
                          [np.array([1.6, 1.6, 0.4]), 4, -0.00036299],
                          [np.array([1.6, 1.6, 0.4]), 6, -0.00159263],
                          [np.array([4.0, 4.0, 1.0]), 4, -0.00263848],
                          [np.array([4.8, 4.8, 1.2]), 4, -0.0040378]])

        obj_hc = hc.HardCore()
        for i in range(len(param)):
            free_energy = obj_hc.free_energy_hc_excess(param[i, 0], param[i, 1])
            # testing params up to a precision of 10^-8
            test = np.allclose(free_energy, param[i, 2], 0, 1e-8)
            self.assertTrue(test)

    def test_pot_chem_hc_excess(self):
        # parameters in [[concentration of the different species (M)], ion size (A), [pot chem excess]]
        param = np.array([[np.array([2.4, 2.4, 0.6]), 4, [0.53372128, 0.53372128, 0.80445306]],
                          [np.array([2.4, 2.4, 0.6]), 6, [4.71590808, 4.71590808, 7.91555483]],
                          [np.array([1.6, 1.6, 0.4]), 4, [0.33147242, 0.33147242, 0.49551186]],
                          [np.array([1.6, 1.6, 0.4]), 6, [1.71084249, 1.71084249, 2.68706435]],
                          [np.array([4.0, 4.0, 1.0]), 4, [1.05121781, 1.05121781, 1.61562475]],
                          [np.array([4.8, 4.8, 1.2]), 4, [1.39524554, 1.39524554, 2.1696644]]])

        obj_hc = hc.HardCore()
        for i in range(len(param)):
            for j in range (3):
                pot_chem = obj_hc.pot_chem_hc_excess(param[i, 0], param[i, 1])
                # testing params up to a precision of 10^-8
                test = np.allclose(pot_chem[j], param[i, 2][j], 0, 1e-8)
                self.assertTrue(test)

    def test_pressure_hc_excess(self):
        # parameters in [[concentration of the different species (M)], ion size (A), pressure excess]
        param = np.array([[np.array([2.4, 2.4, 0.6]), 4, 0.00423018],
                          [np.array([2.4, 2.4, 0.6]), 6, 0.01481318],
                          [np.array([1.6, 1.6, 0.4]), 4, 0.00256312],
                          [np.array([1.6, 1.6, 0.4]), 6, 0.00451955],
                          [np.array([4.0, 4.0, 1.0]), 4, 0.00881886],
                          [np.array([4.8, 4.8, 1.2]), 4, 0.01210029]])

        obj_hc = hc.HardCore()
        for i in range(len(param)):
            press = obj_hc.pressure_hc_excess(param[i, 0], param[i, 1])
            # testing params up to a precision of 10^-8
            test = np.allclose(press, param[i, 2], 0, 1e-8)
            self.assertTrue(test)
