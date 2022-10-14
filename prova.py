# -*- coding: utf-8 -*
# Copyright (C) 2022 s.deastis@studenti.unipi.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
import matplotlib.pyplot as plt
import numpy as np
from ass2_distribuzioni import ProbabilityDensityFunction
if sys.flags.interactive:
    plt.ion()

def test_uniform():
    """
    """
    x = np.linspace(0., 1., 100)
    y = np.full(x.shape, 1.)
    pdf = ProbabilityDensityFunction(x, y)

    return pdf.prob(0.25, 0.75)

if __name__ == '__main__':
    print (test_uniform())
    plt.figure('1')
    x=np.linspace(1.,3.,10)
    y=np.full(x.shape,1)
    rnd= pdf.rnd(1000000)
    plt.plot(x,y)
    plt.figure(2)
    plt.hist(rnd, bins=200)
    plt.show()
