"""
:module: units_unit_test
:platform: Unix, Windows, OS
:synopsis: unit test for units

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""
import unittest
import numpy as np
import aqpolypy.units.units as un


class TestUnits(unittest.TestCase):

    # Testing Boltzman constant
    def test_k_boltzmann(self):
        # boltzmann = 1.380649e-23
        self.assertEqual(un.k_boltzmann(), 1.380649e-23)

    # Testing avogadro
    def test_avogadro(self):
        # avogadro = 6.0221365e23
        self.assertEqual(un.avogadro(), 6.0221365e23)

    # Testing r gas
    def test_r_gas(self):
        # r gas constant = boltzmann * avogadro
        self.assertEqual(un.r_gas(), un.k_boltzmann() * un.avogadro())

    # Testing atm to pascal & atm to bar
    def test_atm_conversion(self):
        # 1 atm = 101325 Pa
        self.assertEqual(un.atm_2_pascal(1), 101325)
        # 1 atm = 1.01325 bar
        self.assertEqual(un.atm_2_bar(1), 1.01325)

    # Testing celsius to kelvin conversion
    def test_kelvin_conversion(self):
        # 0 C = 273.15 K
        self.assertEqual(un.celsius_2_kelvin(0), 273.15)

    # Testing meter to angstrom conversion
    def test_meter_conversion(self):
        # 1m = 10^10 Angstrom
        self.assertEqual(un.m_2_angstrom(1), 1e10)

    # Testing mol/litre to molecule/A^3 conversion
    def test_mol_lit_conversion(self):
        # 1 Mol/litre = to Molecule /A^3
        self.assertEqual(un.mol_lit_2_mol_angstrom(1), un.avogadro() / (1e-3 * un.m_2_angstrom(1) ** 3))

    # Testing 1/4pieps0, e charge, e charge squared
    def test_electrostatic_constants(self):
        self.assertEqual(un.one_over4pi_epsilon0(), 1 / (4 * np.pi * 8.8541878128e-12))
        self.assertEqual(un.e_charge(), 1.60218e-19)
        self.assertEqual(un.e_square(), un.one_over4pi_epsilon0() * (un.e_charge()) ** 2)


if __name__ == '__main__':
    unittest.main()
