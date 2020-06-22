"""
:module: units
:platform: Unix, Windows, OS
:synopsis: Relevant constants and functions used for unit conversion

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - describe changes
"""


import numpy as np


def k_boltzmann():
    """
    Boltzmann constant

    :return: Boltzmann constant (SI units)
    :rtype: float
    """
    return 1.380649e-23


def avogadro():
    """
    Avogadro number

    :return: Avogadro number
    :rtype: float
    """
    return 6.0221365e23


def r_gas():
    """
    Ideal gas constant

    :return: Ideal gas constant R (SI units)
    :rtype: float
    """
    return avogadro() * k_boltzmann()


def atm_2_pascal(p):
    """
    Conversion of atmosphere to pascal

    :param p: pressure in atmospheres
    :return: pressure in pascals
    :rtype: float
    """
    return 101325.0 * p


def atm_2_bar(p):
    """
    Conversion of atmosphere to bar

    :param p: pressure in atmospheres
    :return: pressure in bar
    :rtype: float
    """
    return atm_2_pascal(p) * 1e-5


def celsius_2_kelvin(t):
    """
    Conversion of celsius to Kelvin

    :param t: temperature in celsius
    :return: temperature in Kelvin
    :rtype: float
    """
    return 273.15 + t


def m_2_angstrom(x):
    """
    Conversion from meters to Angstrom

    :param x: length in meters
    :return: length in angstroms
    :rtype: float
    """
    return 1e10 * x


def mol_angstrom_2_mol_mcube(c):
    """

    conversion from molecules angstrom cube to mols meter cube

    :param c: concentration in molecules/Angstrom\ :sup:`3`
    :return: concentration in mols/m :sup:`3`
    """

    return 1e3*c/mol_lit_2_mol_angstrom(1)


def mol_lit_2_mol_angstrom(c):
    """
    Conversion from mols/litre to molecules/Angstrom\ :sup:`3`

    :param c: concentration in mols/litre
    :return: concentration in molecules/Angstrom\ :sup:`3`
    :rtype: float
    """
    # 1l in Angstrom^3
    vol_unit = 1e-3 * (m_2_angstrom(1)) ** 3
    # concentration in molecules/Angstrom^3
    conv = (c * avogadro()) / vol_unit

    return conv


def molality_2_molarity(c, vol_solvent, vol_solute, m_solvent):
    """
    Conversion from molality to molarity conversion

    :param c: molality in SI units
    :param vol_solvent: molar volume of solvent in SI units
    :param vol_solute: molar volume of solute in SI units
    :param: m_solvent: molecular weight of the solvent in SI units
    :return: molarity (SI units)
    :rtype: float
    """

    y = 1e3 / (c * m_solvent)

    return 1e-3 / (vol_solvent * y + vol_solute)


def molarity_2_molality(c, vol_solvent, vol_solute, m_solvent):
    """
    Conversion from molarity to molality conversion

    :param c: molarity in SI units
    :param vol_solvent: molar volume of solvent in SI units
    :param vol_solute: molar volume of solute in SI units
    :param: m_solvent: molecular weight of the solvent in SI units
    :return: molality (SI units)
    :rtype: float
    """

    y = (1e-3 / c - vol_solute) / vol_solvent

    return 1e3 / (y * m_solvent)


def one_over4pi_epsilon0():
    """
    Value of :math:`\\frac{1}{4\\pi\\varepsilon_0}`

    :return: Electrostatic constant (SI units)
    :rtype: float
    """
    return 1 / (4 * np.pi * 8.8541878128e-12)


def e_charge():
    """
    Electron charge

    :return: Electron charge (SI units)
    :rtype: float
    """
    return 1.60218e-19


def e_square():
    """
    Value of :math:`\\frac{e^2}{4\\pi\\varepsilon_0}`

    :return: Value of constant (SI units)
    :rtype: float
    """
    return one_over4pi_epsilon0() * (e_charge()) ** 2
