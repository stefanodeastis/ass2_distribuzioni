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

"""Second assignment for the CMEPDA course, 2022/23.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
MEAN=0
DEVSTD=1
class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    '''Class describing a probability density function.

    Parameters
    ----------
    x : array-like
        The array of x values to be passed to the pdf, assumed to be sorted.
    y : array-like
        The array of y values to be passed to the pdf.
    k : int
        The order of the splines to be created.
    '''
    def __init__(self,x,y,k=3):
        '''Constructor
        '''
        # Normalize the pdf, if it is not.
        norm = InterpolatedUnivariateSpline(x, y, k=k).integral(x[0], x[-1])
        y /= norm
        super().__init__(x, y, k=k)
        ycdf=np.array([self.integral(x[0], temp) for temp in x])
        self.cdf=InterpolatedUnivariateSpline(x,ycdf,k=k)
        # Need to make sure that the vector I am passing to the ppf spline as
        # the x values has no duplicates---and need to filter the y
        # accordingly.
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf,k=k)
    def prob(self, x1, x2):
        """Return the probability for the random variable to be included
        between x1 and x2.
        Parameters
        ----------
        x1: float or array-like
            The left bound for the integration.
        x2: float or array-like
            The right bound for the integration.
        """
        return self.cdf(x2) - self.cdf(x1)
    def rnd(self, size=1000):
        """Return an array of random values from the pdf.
        Parameters
        ----------
        size: int
            The number of random numbers to extract.
        """
        return self.ppf(np.random.uniform(size=size))

def norm_gauss(x,mean,devstd):
    return np.e**(-(x-mean)**2/2*devstd**2)/np.sqrt(2*np.pi*devstd)

if __name__ == '__main__':
    '''
    x=np.linspace(MEAN-5*DEVSTD,MEAN+5*DEVSTD,100)
    y=norm_gauss(x,MEAN,DEVSTD)
    plt.figure(1)
    plt.plot(x,y,'o')
    pdf=ProbabilityDensityFunction(x, y)
    rnd= pdf.rnd(1000000)
    plt.plot(x,y)
    plt.figure(2)
    plt.hist(rnd, bins=200)
    '''
    plt.figure(3)
    x = np.linspace(0., 1., 50000)
    y = np.zeros(x.shape)
    y[x <= 0.5] = 2. * x[x <= 0.5]
    y[x > 0.75] = 3.
    pdf=ProbabilityDensityFunction(x, y,1)
    rnd= pdf.rnd(10000)
    plt.plot(x,y)
    plt.plot(x,pdf.cdf(x))
    plt.plot(x,pdf.ppf(x))
    plt.figure(4)
    plt.hist(rnd, bins=200)
    plt.show()
