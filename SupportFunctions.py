"""
A few support funcitons to help calculate the delay time distribution
Includes: 
* functions to calculate the representative SFR needed to produce a certain COMPAS simulation (from the COMPAS CosmicIntegration suite).
* function to easily read COMPAS data

"""


import numpy as np
import h5py as h5

from astropy.table import Table, Column
import astropy.units as u
from astropy import constants as const

import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u

from matplotlib import ticker, cm
import os
from scipy import stats





def CDF_IMF(m, m1=0.01, m2=0.08, m3=0.5, m4=200.0, a12=0.3, a23=1.3, a34=2.3):
    """
        Calculate the fraction of stellar mass between 0 and m for a three part broken power law.
        Default values follow Kroupa (2001)
            F(m) ~ int_0^m zeta(m) dm

        Args:
            m       --> [float, list of floats] mass or masses at which to evaluate
            mi      --> [float]                 masses at which to transition the slope
            aij     --> [float]                 slope of the IMF between mi and mj

        Returns:
            zeta(m) --> [float, list of floats] value or values of the IMF at m

        NOTE: this is implemented recursively, probably not the most efficient if you're using this
                intensively but I'm not and it looks prettier so I'm being lazy ¯\_(�~C~D)_/¯
    """

    # calculate normalisation constants that ensure the IMF is continuous
    b1 = 1 / (
                (m2**(1 - a12) - m1**(1 - a12)) / (1 - a12) \
                + m2**(-(a12 - a23)) * (m3**(1 - a23) - m2**(1 - a23)) / (1 - a23) \
                + m2**(-(a12 - a23)) * m3**(-(a23 - a34)) * (m4**(1 - a34) - m3**(1 - a34)) / (1 - a34)
                )
    b2 = b1 * m2**(-(a12 - a23))
    b3 = b2 * m3**(-(a23 - a34))

    if isinstance(m, float):
        if m <= m1:
            return 0
        elif m <= m2:
            return b1 / (1 - a12) * (m**(1 - a12) - m1**(1 - a12))
        elif m <= m3:
            return CDF_IMF(m2) + b2 / (1 - a23) * (m**(1 - a23) - m2**(1 - a23))
        elif m <= m4:
            return CDF_IMF(m3) + b3 / (1 - a34) * (m**(1 - a34) - m3**(1 - a34))
        else:
            return 0
    else:
        CDF = np.zeros(len(m))
        CDF[np.logical_and(m >= m1, m < m2)] = b1 / (1 - a12) * (m[np.logical_and(m >= m1, m < m2)]**(1 - a12) - m1**(1 - a12))
        CDF[np.logical_and(m >= m2, m < m3)] = CDF_IMF(m2) + b2 / (1 - a23) * (m[np.logical_and(m >= m2, m < m3)]**(1 - a23) - m2**(1 - a23))
        CDF[np.logical_and(m >= m3, m < m4)] = CDF_IMF(m3) + b3 / (1 - a34) * (m[np.logical_and(m >= m3, m < m4)]**(1 - a34) - m3**(1 - a34))
        CDF[m >= m4] = np.ones(len(m[m >= m4]))
        return CDF




def inverse_CDF_IMF(U, m1=0.01, m2=0.08, m3=0.5, m4=200, a12=0.3, a23=1.3, a34=2.3):
    """
        Calculate the inverse CDF for a three part broken power law.
        Default values follow Kroupa (2001)

        Args:
            U       --> [float, list of floats] A uniform random variable on [0, 1]
            mi      --> [float]                 masses at which to transition the slope
            aij     --> [float]                 slope of the IMF between mi and mj

        Returns:
            zeta(m) --> [float, list of floats] value or values of the IMF at m

        NOTE: this is implemented recursively, probably not the most efficient if you're using this intensively but I'm not so I'm being lazy ¯\_(�~C~D)_/¯
    """
    # calculate normalisation constants that ensure the IMF is continuous
    b1 = 1 / (
                (m2**(1 - a12) - m1**(1 - a12)) / (1 - a12) \
                + m2**(-(a12 - a23)) * (m3**(1 - a23) - m2**(1 - a23)) / (1 - a23) \
                + m2**(-(a12 - a23)) * m3**(-(a23 - a34)) * (m4**(1 - a34) - m3**(1 - a34)) / (1 - a34)
                )
    b2 = b1 * m2**(-(a12 - a23))
    b3 = b2 * m3**(-(a23 - a34))

    # find the probabilities at which the gradient changes
    F1, F2, F3, F4 = CDF_IMF(np.array([m1, m2, m3, m4]), m1=0.01, m2=0.08, m3=0.5, m4=200, a12=0.3, a23=1.3, a34=2.3)
    
    masses = np.zeros(len(U))
    masses[np.logical_and(U > F1, U <= F2)] = np.power((1 - a12) / b1 * (U[np.logical_and(U > F1, U <= F2)] - F1) + m1**(1 - a12), 1 / (1 - a12))
    masses[np.logical_and(U > F2, U <= F3)] = np.power((1 - a23) / b2 * (U[np.logical_and(U > F2, U <= F3)] - F2) + m2**(1 - a23), 1 / (1 - a23))
    masses[np.logical_and(U > F3, U <= F4)] = np.power((1 - a34) / b3 * (U[np.logical_and(U > F3, U <= F4)] - F3) + m3**(1 - a34), 1 / (1 - a34))
    

    return masses


## Calculate the average SF mass that would have been required to evolve one binary
def find_star_forming_mass_per_binary_sampling(m1=0.01, m2=0.08, m3=0.5, m4=200.0, a12=0.3, a23=1.3, a34=2.3,
                                               primary_mass_inverse_CDF=None, mass_ratio_inverse_CDF=None, SAMPLES=20000000, 
                                               binaryFraction = 0.7, Mlower = 10.* u.Msun, Mupper = 150 * u.Msun, m2_min = 0.1 * u.Msun):
    """
        Calculate the star forming mass evolved for each binary in the file.
        This function does this by sampling from the IMF and mass ratio distributions

        Args:
            mi                       --> [float]    masses at which to transition the slope of the IMF (ignored if primary_mass_inverse_CDF is not None)
            aij                      --> [float]    slope of the IMF between mi and mj (ignored if primary_mass_inverse_CDF is not None)
            primary_mass_inverse_CDF --> [function] a function that computes the inverse CDF functoin for the primary mass distribution
                                                    this defaults to the Kroupa IMF (which can be varied using mi, aij)
            mass_ratio_inverse_CDF   --> [function] a function that computes the inverse CDF function for the mass ratio distribution
                                                    this defaults to assuming a uniform mass ratio on [0, 1]
            SAMPLES                  --> [int]      number of samples to draw when creating a mock universe
            binaryFraction           --> [int]      Asusmed binary fraction, default = 0.7
            Mlower                   --> [int]      Minimum primary mass sampled by COMPAS default = 10 
            Mupper                   --> [int]      Maximum primary mass sampled by COMPAS default = 150
            m2_min                  --> [int]      Minimum secondary mass sampled by COMPAS default = 0.1
    """
    # if primary mass inverse CDF is None, assume the Kroupa IMF
    if primary_mass_inverse_CDF is None:
        primary_mass_inverse_CDF = lambda U: inverse_CDF_IMF(U, m1=m1, m2=m2, m3=m3, m4=m4, a12=a12, a23=a23, a34=a34)

    # if mass ratio inverse CDF function is None, assume uniform
    if mass_ratio_inverse_CDF is None:
        mass_ratio_inverse_CDF = lambda q: q

    # randomly sample a large number of masses from IMF, mass ratios from supplied function, binary for boolean
    primary_mass = primary_mass_inverse_CDF(np.random.rand(SAMPLES)) * u.Msun
    mass_ratio = mass_ratio_inverse_CDF(np.random.rand(SAMPLES))
    binary = np.random.rand(SAMPLES)

    # only fbin fraction of stars have a secondary (in a binary)
    binary_mask = binary < binaryFraction

    # assign each a random secondary mass, default 0 because single stars have m2=0 (surprisingly :P)
    secondary_mass = np.zeros(SAMPLES) * u.Msun
    secondary_mass[binary_mask] = primary_mass[binary_mask] * mass_ratio[binary_mask]

    # find the total mass of the whole population
    total_mass = np.sum(primary_mass) + np.sum(secondary_mass)

    # apply the COMPAS cuts on primary and secondary mass
    primary_mask = np.logical_and(primary_mass >= Mlower, primary_mass <= Mupper)
    secondary_mask = secondary_mass > m2_min
    full_mask = np.logical_and(primary_mask, secondary_mask)

    # find the total mass with COMPAS cuts
    total_mass_COMPAS = np.sum(primary_mass[full_mask]) + np.sum(secondary_mass[full_mask])

    # use the totals to find the ratio and return the average mass as well
    f_mass_sampled = total_mass_COMPAS / total_mass
    average_mass_COMPAS = total_mass_COMPAS / len(primary_mass[full_mask])

    # find the average star forming mass evolved per binary in the Universe
    mass_evolved_per_binary = average_mass_COMPAS / f_mass_sampled
    
    return mass_evolved_per_binary


# Read your data
####################################################
## Location of your data
# Rate selects wich mergers you are interested in, using a specific SFRD
def get_essential_data(File_location = ''): 
    print(File_location)
    ####################################################
    '''
    The Bool "DCO_mask" filters for BBHs:  
    1. with an inspiral time that is less than the age of the Universe
    2. excludes systems that experienced a CE from a HG donor (i.e. the flag `Optimistic_CE == False`)
    3. excludes systems that experienced RLOF immediately following a CE (i.e. the flag `Immediate_RLOF>CE == False`)

    In other words, we treat 2. and 3. as stellar mergers and exclude them from the rest of our analysis

    Lastly, we select merging BBHs using the `DCO_mask`, and aditionally exclude systems that evolve Chemically homogeneous. 

    '''
    DCOkey, SYSkey, cecount, dcokey = 'DoubleCompactObjects', 'SystemParameters', 'CE_Event_Count', 'DCOmask'
    ################################################
    ## Essential data for this plot
    ## Open hdf5 file and read relevant columns
    File        = h5.File(File_location ,'r')

    DCO = Table()
    
    try:
        DCO['SEED']                  = File[DCOkey]['SEED'][()] 
    except:
        DCOkey, SYSkey, cecount, dcokey = 'BSE_Double_Compact_Objects', 'BSE_System_Parameters', 'CE_Event_Counter', 'DCOmask'
        DCO['SEED']                  = File[DCOkey]['SEED'][()] 
        
    DCO['Metallicity@ZAMS(1)']   = File[DCOkey]['Metallicity@ZAMS(1)'][()] 
    DCO['CE_Event_Count']        = File[DCOkey][cecount][()] 
    DCO['M_moreMassive']         = np.maximum(File[DCOkey]['Mass(1)'][()], File[DCOkey]['Mass(2)'][()])
    DCO['M_lessMassive']         = np.minimum(File[DCOkey]['Mass(1)'][()], File[DCOkey]['Mass(2)'][()])
    DCO['q_final']               = DCO['M_lessMassive']/DCO['M_moreMassive']
    DCO['mixture_weight']        = File[DCOkey]['mixture_weight'][()]

    DCO['Coalescence_Time']      = File[DCOkey]['Coalescence_Time'][()] 
    DCO['Time']                  = File[DCOkey]['Time'][()] #Myr
    DCO['tDelay']                = DCO['Coalescence_Time'] + DCO['Time'] #Myr

    SYS_DCO_seeds_bool           = np.in1d(File[SYSkey]['SEED'][()], DCO['SEED']) #Bool to point SYS to DCO
    DCO['Stellar_Type@ZAMS(1)']  = File[SYSkey]['Stellar_Type@ZAMS(1)'][SYS_DCO_seeds_bool]
    
    ######### DCO mask by hand to make sure it point to what you want
    DCO['Stellar_Type(1)']    = File[DCOkey]['Stellar_Type(1)'][()]
    DCO['Stellar_Type(2)']    = File[DCOkey]['Stellar_Type(2)'][()]
    BBH_bool      = np.logical_and(DCO['Stellar_Type(1)'] == 14,DCO['Stellar_Type(2)'] == 14 )
    
    DCO['Immediate_RLOF>CE']  = File[DCOkey]['Immediate_RLOF>CE'][()]
    DCO['Optimistic_CE']      = File[DCOkey]['Optimistic_CE'][()]
    DCO['Merges_Hubble_Time'] = File[DCOkey]['Merges_Hubble_Time'][()]
    
    DCO_mask                  = BBH_bool * (DCO['Immediate_RLOF>CE'] == False) * (DCO['Optimistic_CE'] == False) * (DCO['Merges_Hubble_Time'] == True)

    ################################################
    # Compute average stellar mass needed to reproduce this simulation
    ################################################
    n_systems = len(File[SYSkey]['SEED'][()])
    M1_min    = min(File[SYSkey]['Mass@ZAMS(1)'][()]) #minimum ZAMS mass simulated
    mass_evolved_per_binary = find_star_forming_mass_per_binary_sampling(binaryFraction = 0.7, Mlower = M1_min* u.Msun, Mupper = 150 * u.Msun, m2_min = 0.1 * u.Msun)
    Average_SF_mass_needed = (mass_evolved_per_binary * n_systems)
    print('Average_SF_mass_needed', Average_SF_mass_needed)
    
    ############
    File.close()

    ################################################
    # Bools to select merging BBHs w.o. CHE only
    nonCHE_bool         = DCO['Stellar_Type@ZAMS(1)'] != 16
    rate_nonCHE_bool    = DCO['Stellar_Type@ZAMS(1)'][DCO_mask] != 16

    # Filter both the BBH table and the intrinsic rate data
    merging_BBH         = DCO[DCO_mask * nonCHE_bool]

    return merging_BBH, Average_SF_mass_needed #Red_intr_rate_dens



#############################################################################################################
# Get the number of certain event per metallicity bin
# We want to count the number of events (i.e. unstable mass transfer, stellar merger, CE ejection) per metallicity bin
#############################################################################################################
def get_numbers(data_dir = '', simname = 'faccTHERMALzetaHG6.0RemMassFRYER2012SNDELAYED', 
                keys_of_interest = ['merging_BBH', 'merging_NSBH', 'merging_NSNS', 'Stellar_mergers', 'CE_Event_Counter', 'EjectedCE'], verbose = False):
    """
    data_dir = proj_dir + '/v02.26.03/N1e7Grid_BBH_BHNS_optimized/', 
    simname = 'faccTHERMALzetaHG6.0RemMassFRYER2012SNDELAYED', 
    keys_of_interest = ['merging_BBH', 'merging_NSBH', 'merging_NSNS', 'Stellar_mergers', 'CE_Event_Counter', 'EjectedCE']
    verbose = False
    """

    ################################################
    # Count the number of events 
    ################################################
    # Read data and put in astropy table
    File        = h5.File(data_dir + simname +'/output/COMPAS_Output_wWeights.h5' ,'r')
    if verbose: print(File['BSE_System_Parameters'].keys())

    # Add relevant SYS keys to SYS_num table
    SYS_nums = Table()
    for key in ['SEED', 'mixture_weight', 'Metallicity@ZAMS(1)', 'CE_Event_Counter', 'Merger', 'Optimistic_CE', 'Immediate_RLOF>CE']:
        SYS_nums[key] = File['BSE_System_Parameters'][key][()]
        
    # General numbers for this simulation
    n_systems = len(SYS_nums['SEED'])
    M1_min    = min(File['BSE_System_Parameters']['Mass@ZAMS(1)'][()]) #minimum ZAMS mass simulated

    #Bool to point SYS to DCO
    SYS_DCO_seeds_bool           = np.in1d(File['BSE_System_Parameters']['SEED'][()], File['BSE_Double_Compact_Objects']['SEED'][()])

    # Add relevant DCO keys to SYS_num table
    for dcokey in ['Stellar_Type(1)', 'Stellar_Type(2)', 'Merges_Hubble_Time']:
        SYS_nums[dcokey]  = np.zeros(len(SYS_nums))
        # Thoses systems that become DCO
        SYS_nums[dcokey][SYS_DCO_seeds_bool] = File['BSE_Double_Compact_Objects'][dcokey][()]

    File.close()

    ################################################
    # Create a new table events per metallicity bin
    ################################################
    events_per_Zbin = Table()
    
    ################################################
    # Create metallicity bins
    ################################################
    Metal_bins  = np.linspace(-4, max(np.log10(SYS_nums['Metallicity@ZAMS(1)'])), 15)
    center_bins = (Metal_bins[1:] + Metal_bins[:-1])/2.    
    binwidts    = np.diff(Metal_bins)
    events_per_Zbin['Metallicity_bin'] = center_bins
    events_per_Zbin['dlnZ']            = binwidts
    
    # Digitize your systems per metallicity
    SYSbin_indices = np.digitize(np.log10(SYS_nums['Metallicity@ZAMS(1)']) , Metal_bins, right=True)  
    
    # AIS fraction per metal bin normalized
    events_per_Zbin['mixture_weight_sum']  = np.bincount(SYSbin_indices, weights=SYS_nums['mixture_weight'])[1:]/sum(SYS_nums['mixture_weight'])
    
    ###########
    # Number of BBH mergers
    if 'merging_BBH' in keys_of_interest:
        BBH_bool      = np.logical_and(SYS_nums['Stellar_Type(1)'] == 14,SYS_nums['Stellar_Type(2)'] == 14 )
        NO_RLOF_CE    = SYS_nums['Immediate_RLOF>CE'] == False
        pessimisticCE = SYS_nums['Optimistic_CE'] == False
        merger        = SYS_nums['Merges_Hubble_Time'] == True
        stable        = SYS_nums['CE_Event_Counter'] == 0
        #
        SYS_nums['merging_BBH']                 = BBH_bool * NO_RLOF_CE * pessimisticCE * merger
        events_per_Zbin['merging_BBH']          = np.bincount(SYSbin_indices, weights=SYS_nums['merging_BBH']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
        
        # Split by stable only or CE channel
        SYS_nums['merging_BBH_stable']          = BBH_bool * NO_RLOF_CE * pessimisticCE * merger * stable
        events_per_Zbin['merging_BBH_stable']   = np.bincount(SYSbin_indices, weights=SYS_nums['merging_BBH_stable']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
        
        SYS_nums['merging_BBH_CE']              = BBH_bool * NO_RLOF_CE * pessimisticCE * merger  *np.invert(stable)
        events_per_Zbin['merging_BBH_CE']       = np.bincount(SYSbin_indices, weights=SYS_nums['merging_BBH_CE']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
        

    ###########
    # Number of BHNS mergers
    if 'merging_NSBH' in keys_of_interest:
        NSBH_bool     = np.logical_or(np.logical_and(SYS_nums['Stellar_Type(1)'] == 13,SYS_nums['Stellar_Type(2)'] == 14 ),
                                      np.logical_and(SYS_nums['Stellar_Type(1)'] == 14,SYS_nums['Stellar_Type(2)'] == 13 ))
        SYS_nums['merging_NSBH']          = NSBH_bool * NO_RLOF_CE * pessimisticCE * merger
        events_per_Zbin['merging_NSBH']   = np.bincount(SYSbin_indices, weights=SYS_nums['merging_NSBH']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
        
    ###########
    # Number of NSNS mergers
    if 'merging_NSNS' in keys_of_interest:
        NSNS_bool     = np.logical_and(SYS_nums['Stellar_Type(1)'] == 13,SYS_nums['Stellar_Type(2)'] == 13 )
        SYS_nums['merging_NSNS']         = NSNS_bool * NO_RLOF_CE * pessimisticCE * merger
        events_per_Zbin['merging_NSNS']  = np.bincount(SYSbin_indices, weights=SYS_nums['merging_NSNS']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
        
        
    ###########
    # Number of CE that occurred in all systems
    if 'CE_Event_Counter' in keys_of_interest:
        events_per_Zbin['CE_Event_Counter']    = np.bincount(SYSbin_indices, weights=SYS_nums['CE_Event_Counter']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]
    
    ###########
    # Number of stellar mergers
    if 'Stellar_mergers' in keys_of_interest:
        # Wherever either of these is true, the CE should be counted as a stellar merger
        SYS_nums['Merger'][SYS_nums['Optimistic_CE'] == True ] = 1
        SYS_nums['Merger'][SYS_nums['Immediate_RLOF>CE'] == True ] = 1

        # OK weirdly, there are stellar mergers that happen when no CE has happened, 
        # For now I am going to disregard those mergers (not count them)
        merged_wo_CE = np.where(SYS_nums['Merger'] >  SYS_nums['CE_Event_Counter'])[0]
        SYS_nums['Merger'][merged_wo_CE ] = 0
        events_per_Zbin['Stellar_mergers']     = np.bincount(SYSbin_indices, weights=SYS_nums['Merger']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]

    ###########
    # Number of Succesfull ejections is all CE - stellar mergers
    if 'EjectedCE' in keys_of_interest:
        SYS_nums['EjectedCE']        = SYS_nums['CE_Event_Counter'] - SYS_nums['Merger']
        events_per_Zbin['EjectedCE'] = np.bincount(SYSbin_indices, weights=SYS_nums['EjectedCE']*SYS_nums['mixture_weight'])[1:]# Bincount counts the 0's, omit them with [1:]

    ################################################
    # Compute average stellar mass needed to reproduce this simulation
    ################################################
    mass_evolved_per_binary = find_star_forming_mass_per_binary_sampling(binaryFraction = 0.7, Mlower = M1_min* u.Msun, Mupper = 150 * u.Msun, m2_min = 0.1 * u.Msun)

    Average_SF_mass_needed = (mass_evolved_per_binary * n_systems)
    if verbose: print('Average_SF_mass_needed', Average_SF_mass_needed)
        
    events_per_Zbin['Average_SF_mass_needed'] = Average_SF_mass_needed

        
    return events_per_Zbin
    



