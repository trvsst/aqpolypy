"""
:module: units_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for concentration

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added unit test for molality to molarity conversion.
..                  - Added unit test for molarity to molality conversion.
"""

import unittest
import numpy as np
import aqpolypy.units.concentration as con
import aqpolypy.water.WaterMilleroBP as wfm
import aqpolypy.salt.SaltNaClRP as nacl


class TestUnits(unittest.TestCase):
    # Testing molality to molarity conversion
    def test_molality_conversion(self):
        # 1 molal NaCl = 0.9790370012 molar NaCl at 25C
        c = 1
        water = wfm.WaterPropertiesFineMillero(298.15)
        salt = nacl.NaClPropertiesRogersPitzer(298.15)
        convert = con.molality_2_molarity(c, water.molar_volume(), salt.molar_vol(c), water.MolecularWeight)
        # test precision up to 10^-6
        test = np.allclose(convert, 0.9790370012, 0, 1e-6)
        self.assertTrue(test)

    # Testing molarity to molality conversion
    def test_molarity_conversion(self):
        # 1 molar NaCl = 1.021411855 molar NaCl at 25C
        c = 1
        water = wfm.WaterPropertiesFineMillero(298.15)
        salt = nacl.NaClPropertiesRogersPitzer(298.15)
        convert = con.molarity_2_molality(c, water.molar_volume(), salt.molar_vol(1.021411855), water.MolecularWeight)
        # test precision up to 10^-6
        test = np.allclose(convert, 1.021411855, 0, 1e-2)
        self.assertTrue(test)
