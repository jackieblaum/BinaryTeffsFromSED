import numpy as np
from astropy.modeling import models, fitting
from scipy.optimize import minimize
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plt
from pystellibs import BaSeL, Kurucz
from extinction import ccm89, apply,remove
import pyphot
from pyphot import unit

class Simulator:

    def __init__(self, filters):
        self.filters = filters

        # load kurucz library for SED; Castelli and Kurucz 2004 or ATLAS9
        self.kurucz = Kurucz()

    def create_intrinsic_sed(self, teff,logg,lum,Z,dist):
        
        #Teff in K, lum in Lsun, distance in pc
        dist=dist*3.086e+18 # convert pc to cm
        fluxconv=4*np.pi*dist*dist # to convert luminosity to flux
        ap = (np.log10(teff), logg, np.log10(lum), Z) #input order: logT, logg, logL, Z
        kurucz_sed = self.kurucz.generate_stellar_spectrum(*ap) # from pystellibs
        
        # wavelengths must be determined based on ap
        sed=pd.DataFrame({'wavelength':self.kurucz._wavelength,'flux':kurucz_sed/fluxconv}) 
        
        #Return intrinsic SED of one star at a given distance
        #Returns wavelength in angstroms, flux in erg/cm2/s/AA
        return sed

    def create_apparent_sed(self, teff1,teff2,logg1,logg2,lum1,lum2,Z,dist,A_V,R_V=3.1):
        
        #For star 1 and star 2: Teff in K, lum in Lsun, distance in pc
        #ebv is reddening E(B-V)
        #R_V is the extinction factor, default at 3.1
        sed_1=self.create_intrinsic_sed(teff1,logg1,lum1,Z,dist)
        sed_2=self.create_intrinsic_sed(teff2,logg2,lum2,Z,dist)
        
        comb_sed=sed_1+sed_2
        # Use the Cardelli, Clayton & Mathis (1989) extinction law _V = 3.1, input is A_V and Rv
        # apply and ccm89 are from extinction; 'apply' applies extinction to flux values
        comb_sed['apparent_flux']=apply(ccm89(comb_sed.wavelength.values, A_V, R_V),comb_sed['flux'].values)
        
        #Returns the combined binary SED with reddening
        #Returns wavelength in angstroms, flux in erg/cm2/s/AA
        return comb_sed
        
    def plot_apparent_sed(binary_sed):
        
        #Input: take binary_sed output from create_apparent_sed
        
        fig, ax= plt.subplots(nrows=1, ncols=1,figsize=(12,10))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')  
        ax.tick_params(direction='in',axis='both',which='minor',length=3,width=2,labelsize=18)
        ax.tick_params(direction='in',axis='both',which='major',length=6,width=2,labelsize=18)
        ax.minorticks_on()  
        ax.set_xlabel(r'$ \rm Wavelength [nm]$',fontsize=24)
        ax.set_ylabel(r'$\lambda F_\lambda~[\rm erg/cm^2/s]$',fontsize=24)

        #Truncate the SED for visualization purposes
        binary_sed_plot=binary_sed.loc[binary_sed.wavelength>2000]
        ax.loglog(binary_sed_plot.wavelength/10, binary_sed_plot.flux*binary_sed_plot.wavelength/10, label='Flux',c='k')
        ax.loglog(binary_sed_plot.wavelength/10, binary_sed_plot.apparent_flux*binary_sed_plot.wavelength/10, label='Reddened Flux',c='r')
        ax.legend(fontsize=24)
        plt.show()

if __name__ == '__main__':
    print('Getting filters...')
    filters = pyphot.get_library()
    sim = Simulator(filters)
    for i in range(10):
        # Sample parameters from MIST isochrones for each star
        teff1 = np.random.uniform(3000, 10000)
        teff2 = np.random.uniform(3000, 10000)
        logg1 = np.random.uniform(0, 5)
        logg2 = np.random.uniform(0, 5)
        lum1 = np.random.uniform(0, 5)
        lum2 = np.random.uniform(0, 5)
        Z = np.random.uniform(0, 0.02)
        dist = np.random.uniform(100, 1000)
        A_V = np.random.uniform(0, 1)
        R_V = np.random.uniform(2, 5)

        # Simulate SEDs for each binary
        binary_sed = sim.create_apparent_sed(teff1, teff2, logg1, logg2, lum1, lum2, Z, dist, A_V, R_V)
        binary_sed.to_csv(f'../data/sims/binary_sed_{i}.csv')
        params = np.array([teff1, teff2, logg1, logg2, lum1, lum2, Z, dist, A_V, R_V])
        np.save(f'../data/sims/params_{i}.npy', params)
        sim.plot_apparent_sed(binary_sed)