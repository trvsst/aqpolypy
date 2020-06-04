"""
:module: Units
:platform: Unix, Windows, OS
:synopsis: Relevant constants and functions used for unit conversion

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - describe changes
"""


import numpy as np


def k_boltzmann():
    """
    Boltzmann Constant

    :return: Boltzman constant SI units
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

    :return: Ideal gas constant R
    :rtype: float
    """

    return avogadro() * k_boltzmann()


def atm_2_pascal(p):
    """
    conversion of atmosphere to pascal

    :param p: pressure in atmospheres
    :return: pressure in pascals
    :rtype : float
    """

    return 101325.0 * p


def atm_2_bar(p):
    """
    conversion of atmosphere to bar

    :param p: pressure in atmospheres
    :return: pressure in bar
    :rtype : float
    """

    return atm_2_pascal(p) * 1e-5


def celsius_2_kelvin(t):
    """
    conversion of celsius to kelvin

    :param t: temperature in celsius
    :return: temperature in Kelvin
    :rtype : float
    """

    return 273.15 + t


def m_2_angstrom(x):
    """
    Conversion from meters to Angstrom

    :param x: length in meters
    :return: length in angstroms
    :rtype : float
    """

    return 1e10 * x


def mol_lit_2_mol_angstrom(c):
    """
    Conversion from mols/litre to molecules/Angstrom3

    :param c: concentration in mols/litre
    :return: concentration in molecules/Angstrom3
    :rtype : float
    """

    # 1l in Angstrom^3
    vol_unit = 1e-3 * (m_2_angstrom(1)) ** 3
    # concentration in molecules/Angstrom^3
    conv = (c * avogadro()) / vol_unit

    return conv


def one_over4pi_epsilon0():
    """
    value of math:`\frac{1}{\4\pi\varepsilon_0}`

    :return: electrostatic constant (SI units)
    :rtype: float
    """

    return 1 / (4 * np.pi * 8.8541878128e-12)


def e_charge():
    """
    electron charge

    :return: electron charge (SI units)
    :rtype: float
    """

    return 1.60218e-19


def e_square():
    """
    value of math:`\frac{e^2}{\4\pi\varepsilon_0}`

    :return: value of constant (SI units)
    :rtype: float
    """

    return one_over4pi_epsilon0() * (e_charge()) ** 2
