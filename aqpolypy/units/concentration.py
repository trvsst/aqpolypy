"""
:module: concentration
:platform: Unix, Windows, OS
:synopsis: Relevant functions used for concentration conversion

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history:
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - Added molality to molarity conversion.
..                  - Added molarity to molality conversion.
"""


def molality_2_molarity(c, vol_solvent, vol_solute, m_solvent):
    """
    Conversion from molality to molarity conversion

    :param c: molality in mols_solute / Kg_solvent
    :param vol_solvent: molar volume of solvent in SI units
    :param vol_solute: molar volume of solute in SI units
    :param m_solvent: molecular weight of the solvent
    :return: molarity in mols / L
    :rtype: float
    """

    y = 1e3 / (c * m_solvent)

    return 1e-3 / (vol_solvent * y + vol_solute)


def molarity_2_molality(c, vol_solvent, vol_solute, m_solvent):
    """
    Conversion from molarity to molality conversion

    :param c: molarity in mols / L
    :param vol_solvent: molar volume of solvent in SI units
    :param vol_solute: molar volume of solute in SI units
    :param m_solvent: molecular weight of the solvent
    :return: molality mols_solute / Kg_solvent
    :rtype: float
    """

    y = (1e-3 / c - vol_solute) / vol_solvent

    return 1e3 / (y * m_solvent)
