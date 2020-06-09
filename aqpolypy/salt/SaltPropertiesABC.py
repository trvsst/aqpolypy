"""
:module: SaltPropertiesABC
:platform: Unix, Windows, OS
:synopsis: Abstract class used for deriving child classes

.. moduleauthor:: Alex Travesset <trvsst@ameslab.gov>, May2020
.. history::
..                Kevin Marin <marink2@tcnj.edu>, May2020
..                  - changes
"""

from abc import ABC, abstractmethod


class SaltProperties(ABC):
    def __init__(self):
        """
        constructor

        :param :
        :param :
        :instantiate:

        """

        # Attributes

    @abstractmethod
    def method1(self):
        """
        Abstract method:
        """
        pass

    @abstractmethod
    def method2(self):
        """
        Abstract method:
        """
        pass

    @abstractmethod
    def method3(self):
        """
        Abstract method:
        """
        pass
