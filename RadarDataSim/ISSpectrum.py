"""
Python class to calculate an incoherent scatter spectrum.  Based on ISspectrum.java as written
by John Holt at Millstone Hill in February 1996.

P. Erickson  2009-07-09
$Id$
"""

import math
import numpy

class ISSpectrum:
    """ ISSpectrum is a class which calculates an incoherent scatter spectrum as a function
    of frequency from an input set of physical parameters and a measurement frequency.
    """

    def __init__(self, centerFrequency = 440.2, bMag = 0.4e-4, nspec=64, sampfreq=50e3):
        """ __init__(centerFrequency, bMag) initializes the class.

        Inputs:
           centerFrequency - float: center frequency in MHz.  Default = 440.2.
           bMag - float: Magnetic field strength in Gauss.  Default = 0.4e-4.
           nspec - integer: number of points in spectra.  Default = 64.
           tau - float: sampling frequency in Hz.  Default = 50e3.
        Returns:
           None
        Affects:
           Sets up a number of physical internal constants.
        Exceptions:
           None
        """

        # Fixed physical constants
        self.__alpha = 0.314
        self.__b = bMag
        self.__re = 2.81777e-15
        self.__amu = 1.660e-27
        self.__e = 1.60210e-19
        self.__boltk = 1.38054e-23
        self.__rme = 9.1091e-31
        self.__alpha1 = 1./math.cos(self.__alpha)
        self.__alpha2 = math.sin(self.__alpha)*math.sin(self.__alpha)/2.0
        self.__alpha3 = self.__alpha1*self.__alpha1
        self.__pi1 = math.sqrt(math.pi)
        self.__re1 = self.__re**2/math.pi
        self.__rkfac = 1.61998e6

        # set center frequency-dependent constants
        self.__freq = centerFrequency
        self.__setFreqConstants__()

        # set sampling parameters
        self.__nspec = nspec
        self.__sampfreq = sampfreq

        self.__nspec1D = self.__nspec/2 + 1
        self.__delom = 2*math.pi*self.__sampfreq/self.__nspec

    def __setFreqConstants__(self):
        """ __setFreqConstants__() is a private function which initializes several
        internal constants based on the center frequency.

        Inputs:
            None
        Returns:
            None
        Affects:
            Sets up a number of physical internal constants.
        Exceptions:
            None
        """
        self.__rlam = 2.9979*100.0/self.__freq
        self.__rk = 4.0*math.pi/self.__rlam
        self.__thface = math.sqrt(self.__rme/(2.0*self.__boltk))/self.__rk
        self.__thfaci = math.sqrt(self.__amu/(2.0*self.__boltk))/self.__rk
        self.__phface = (self.__e*self.__b/self.__rme)*math.sqrt(self.__rme/(2.*self.__boltk))/self.__rk
        self.__phfaci = (self.__e*self.__b/self.__amu)*math.sqrt(self.__amu/(2.*self.__boltk))/self.__rk

    def __dfun__(self, tr, pfac):
        """ __dfun(tr, pfac)__ is a private function to calculate the Debye factor corresponding to
        a given temperature ratio and a power factor pfac = 4761. * te * rk**2 / (pnorm*p).
        Newton's method is used to solve dfun*(1+tr+dfun)*(1+dfun)-pfac=0.

        Inputs:
            tr - float: Temperature ratio Te/Ti
            pfac - float: Power factor pfac (see above)
        Returns:
            float: Debye factor.
        Affects:
            None
        Exceptions:
            None
        """

        # initialize
        dfun = 0.0

        for i in range(10):

            a1 = 1. + tr
            a2 = 2.*(2.+tr)
            dfun = pfac/a1
            f = dfun*(a1+dfun)*(1.+dfun) - pfac
            fpr = dfun*(3.*dfun+a2) + a1
            ddf = f/fpr
            dfun = dfun - ddf
            i += 1
            if (ddf < .001):
                break

        return dfun

    def __daw__(self, x):
        """ __daw__(x) is a private function which solves Dawson's integral.

        Inputs:
            x - float: argument to Dawson's integral.
        Returns:
            Dawson's integral evaluated at x
        Affects:
            None
        Exceptions:
            None
        """

        # fixed integration coefficients
        p1 = [2.31569752013e+05, -2.91794643008e+04,
              9.66963981917e+03, -4.35011602076e+02,
              5.46161225567e+01, -8.54106811960e-01,
              2.08468351039e-02]
        q1 = [2.31569752014e+05,  1.25200370319e+05,
              3.13846201382e+04,  4.74470984407e+03,
              4.66849065451e+02,  2.93919956126e+01,
              1.00000000000e+00]
        p2 = [5.01401061170e-01, -7.44990505794e+00,
              7.50778164901e+00, -2.66290010738e+01,
              3.09840878634e+01, -4.08473912127e+01]
        q2 = [1.88975530144e-01,  7.02049807292e+01,
              4.18218063378e+01,  3.73430847283e+01,
              1.25993235468e+03]
        p3 = [5.00001538408e-01, -1.53672069272e+00,
              -1.77068693718e+01,  7.49584016278e+00,
              4.02187490206e+01, -5.93915918500e+01]
        q3 = [2.49811162845e-01, -6.53419359861e-01,
              2.04866410977e+02, -2.29875841929e+00,
              2.53388006964e+03]
        p4 = [5.00000001675e-01, -2.50017116686e+00,
              -4.67312022141e+00, -1.11952164237e+01]
        q4 = [7.49999190567e-01, -2.48787658804e+00,
              -4.12544065608e+00]

        y = x*x

        if (y < 6.25):
            sump = (((((p1[6]*y + p1[5])*y + p1[4])*y + p1[3])*y +
                        p1[2])*y + p1[1])*y + p1[0]
            sumq = (((((q1[6]*y + q1[5])*y + q1[4])*y + q1[3])*y +
                        q1[2])*y +q1[1])*y + q1[0]
            daw = x*sump/sumq
        elif (y < 12.25):
            frac = 0.0;
            for i in range(4, -1, -1):
                frac = q2[i]/(p2[i+1] + y + frac)
            daw = (p2[0] + frac)/x
        elif (y < 25.0):
            frac = 0.0
            for i in range(4, -1, -1):
                frac = q3[i]/(p3[i+1] + y + frac)
            daw = (p3[0] + frac)/x
        elif (y < 1.e12):
            w2 = 1.0/y
            frac = 0.0
            for i in range(2, -1, -1):
                frac = q4[i]/(p4[i+1] + y + frac)
            frac = p4[0] + frac
            daw = (0.5 + 0.5*w2*frac)/x
        else:
            daw = 0.5/x

        return daw

    def __spect2__(self, ti, te, df, p2, rm1, rm2, n, delom):
        """ __spect2__(ti, te, df, p2, rm1, rm2, n, delom) is a private function to calculate
        the incoherent scatter spectrum ion line.

        Inputs:
            ti - float: Ion temperature, K
            te - float: Electron temperature, K
            df - float: Debye factor
            p2 - float: Fractional composition of 2nd ion species
            rm1 - float: Mass of ion species 1 (AMU)
            rm2 - float: Mass of ion species 2 (AMU)
            n - float: Number of frequencies at which to calculate the spectrum
            delom - float: Frequency step (2 * pi * delta_Hz)
        Returns:
            List of floats with IS power spectrum values.
        Affects:
            None
        Exceptions:
            None
        """

        tr = te/ti
        sqrte = 1./math.sqrt(te)
        sqrti = 1./math.sqrt(ti)
        the = self.__thface*sqrte
        thi = self.__thfaci*sqrti
        thi1 = math.sqrt(rm1)*thi
        thi2 = math.sqrt(rm2)*thi
        phi = self.__phface*sqrte
        phifac = 1. - self.__alpha2/(phi*phi)
        fac = phifac*the*self.__alpha1
        p1 = 1. - p2

        omega = 0.0
        sp = []

        for i in range(n):

            # Calculate electron admittance
            om2 = omega*omega

            th = the*omega
            tha = th*self.__alpha1
            exfe = math.exp(-th*th*self.__alpha3)
            yer = self.__pi1*fac*exfe
            yei = 1.0 - 2.0*fac*omega*self.__daw__(tha)

            # Calculate admittance of second ion
            if p2 > 0.0:
                th2 = thi2*omega
                exf2 = math.exp(-th2*th2)
                yir2 = self.__pi1*thi2*exf2
                yii2 = 1.0 - 2.0*th2*self.__daw__(th2)
            else:
                yir2 = 0.0
                yii2 = 0.0

            # Calculate admittance of first ion
            th1 = thi1*omega
            exf1 = math.exp(-th1*th1)
            yir1 = self.__pi1*thi1*exf1
            yii1 = 1.0 - 2.0*th1*self.__daw__(th1)

            # Calculate spectrum
            yir = p1*yir1 + p2*yir2
            yii = p1*yii1 + p2*yii2
            yef = om2*yer*yer + yei*yei;
            yif1 = tr*yir
            yif1 = om2*yif1*yif1
            yif2 = tr*yii + df
            yif2 = yif2*yif2
            yif = yif1 + yif2
            sn = yef*yir + yif*yer
            sd1 = omega*(yer + tr*yir)
            sd1 = sd1*sd1
            sd2 = yei + tr*yii + df
            sd2 = sd2*sd2
            sd = sd1 + sd2
            sp.append(sn/sd)

            # Step frequency forwards
            omega = omega + delom;

        return sp

    def getSpectrum(self, ti, tr, po, rm1, rm2, p2):
        """getSpectrum(ti, tr, po, rm1, rm2, p2) is a public function to calculate and return the incoherent
        scatter ion line power spectrum.

        Inputs:
            ti - float: Ion temperature, K
            tr - float: Ion-to-electron temperature ratio
            po - float: Log10(Electron density, m^-3)
            rm1 - float: Mass of ion species 1 (AMU) - assumed singly charged
            rm2 - float: Mass of ion species 2 (AMU) - assumed singly charged
            p2 - float: Fractional composition of 2nd ion species
        Returns:
            Tuple - (omega, sp)
            omega: List of frequencies for IS power spectrum evaluations.
            sp: List of floats with IS power spectrum values.
        Affects:
            None
        Exceptions:
            None
        """

        # set plasma parameters
        te = tr*ti
        pfac = self.__rkfac*te/math.pow(10.0, po)
        df = self.__dfun__(tr, pfac)

        # compute spectrum (normalized)
        omega = numpy.zeros(self.__nspec, numpy.float)

        sph = self.__spect2__(ti, te, df, p2, rm1, rm2, self.__nspec1D, self.__delom)
        sp = numpy.concatenate((sph[:0:-1], sph))
        sp /= max(sp)
        omega = self.__delom * numpy.arange(-(len(sph)-1), len(sph))
        omega /= 2*math.pi  # result is in Hz, not radians/sec

        return (omega, sp)

    def getSpectrumParameterVector(self, vec):
        """getSpectrum(ti, tr, pi, rm1, rm2, p2) is a public function to calculate and return the incoherent
        scatter ion line power spectrum.

        Inputs:
            vec - list of parameters (ti, tr, po, rm1, rm2, p2)
            ti - float: Ion temperature, K
            tr - float: Ion-to-electron temperature ratio
            po - float: Log10(Electron density, m^-3)
            rm1 - float: Mass of ion species 1 (AMU) - assumed singly charged
            rm2 - float: Mass of ion species 2 (AMU) - assumed singly charged
            p2 - float: Fractional composition of 2nd ion species
        Returns:
            Tuple - (omega, sp)
            omega: List of frequencies for IS power spectrum evaluations.
            sp: List of floats with IS power spectrum values.
        Affects:
            None
        Exceptions:
            None
        """

        return self.getSpectrum(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
