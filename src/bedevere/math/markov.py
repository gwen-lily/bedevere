"""Markov chains and their uses.

###############################################################################
# package:  bedevere                                                          #
# website:  github.com/noahgill409/bedevere                                   #
# email:    noahgill409@gmail.com                                             #
###############################################################################

"""

from dataclasses import dataclass

from bedevere.math.distribution import StochasticMatrix


@dataclass
class MarkovChain(StochasticMatrix):
    """__description__."""
