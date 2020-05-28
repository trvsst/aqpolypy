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


def atm_2_pascal():
    """
    conversion of atmosphere to pascal

    :return: Atmosphere to Pascal conversion
    :rtype: float
    """

    return 101325.0


def atm_2_bar():
    """
    conversion of atmosphere to bar

    :return: Atmosphere to bar conversion
    :rtype: float
    """

    return atm_2_pascal() * 1e-5


def celsius_2_kelvin(t):
    """

    :param t: temperature in celsius
    :return: temperature in Kelvin
    :rtype : float
    """

    return 273.15 + t


def m_2_angstrom():
    """
    Conversion from meters to Angstrom

    :return: conversion factor
    :rtype: float
    """

    return 1e10


def mol_lit_2_mol_angstrom():
    """
    Conversion from mols/litre to molecules/Angstrom3

    :return: conversion factor
    :rtype: float
    """

    # 1l in Angstrom^3
    vol_unit = 1e-3 * (m2angstrom()) ** 3
    # concentration in molecules/Angstrom^3
    conv = avogadro() / vol_unit

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
