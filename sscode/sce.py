#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
# Purpose:
# Dependencies: 	numpy
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#
# https://github.com/stijnvanhoey/Optimization_SCE
#
#-------------------------------------------------------------------------------
#!/usr/bin/env python

## Refer to paper:
##  'EFFECTIVE AND EFFICIENT GLOBAL OPTIMIZATION FOR CONCEPTUAL
##  RAINFALL-RUNOFF MODELS' BY DUAN, Q., S. SOROOSHIAN, AND V.K. GUPTA,
##  WATER RESOURCES RESEARCH, VOL 28(4), PP.1015-1031, 1992.

# basic
import sys
import math

# arrays
import random
import numpy as np
import pandas as pd
import xarray as xr

# plotting
from matplotlib import pyplot as plt

# maths
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# custom
from basic_functions import *


def cceua(s,sf,bl,bu,icall,maxn,iseed,data=None):
    #  This is the subroutine for generating a new point in a simplex
    #
    #   s(.,.) = the sorted simplex in order of increasing function values
    #   s(.)   = function values in increasing order
    #
    # LIST OF LOCAL VARIABLES
    #   sb(.)   = the best point of the simplex
    #   sw(.)   = the worst point of the simplex
    #   w2(.)   = the second worst point of the simplex
    #   fw      = function value of the worst point
    #   ce(.)   = the centroid of the simplex excluding wo
    #   snew(.) = new point generated from the simplex
    #   iviol   = flag indicating if constraints are violated
    #           = 1 , yes
    #           = 0 , no

    nps,nopt=s.shape
    n = nps
    m = nopt
    alpha = 1.0
    beta = 0.5

    # Assign the best and worst points:
    sb=s[0,:]
    fb=sf[0]
    sw=s[-1,:]
    fw=sf[-1]

    # Compute the centroid of the simplex excluding the worst point:
    ce= np.mean(s[:-1,:],axis=0)

    # Attempt a reflection point
    snew = ce + alpha*(ce-sw)

    # Check if is outside the bounds:
    ibound=0
    s1=snew-bl
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=1

    s1=bu-snew
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=2

    if ibound >= 1:
        snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!

    # fnew = functn(nopt,snew); MATLAB code
    fnew = EvalObjF(nopt,snew,data=data)
    icall += 1

    # Reflection failed; now attempt a contraction point:
    if fnew > fw:
        snew = sw + beta*(ce-sw)
        fnew = EvalObjF(nopt,snew,data=data)
        icall += 1

    # Both reflection and contraction have failed, attempt a random point;
        if fnew > fw:
            snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!
            fnew = EvalObjF(nopt,snew,data=data)
            icall += 1

    # END OF CCE
    return snew,fnew,icall


def sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,data=None):
    # This is the subroutine implementing the SCE algorithm,
    # written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
    #
    # Definition:
    #  x0       = the initial parameter array at the start; np.array
    #           = the optimized parameter array at the end;
    #  f0       = the objective function value corresponding to the initial parameters
    #           = the objective function value corresponding to the optimized parameters
    #  bl       = the lower bound of the parameters; np.array
    #  bu       = the upper bound of the parameters; np.array
    #  iseed    = the random seed number (for repetetive testing purpose)
    #  iniflg   = flag for initial parameter array (=1, included it in initial
    #             population; otherwise, not included)
    #  ngs      = number of complexes (sub-populations)
    #  npg      = number of members in a complex
    #  nps      = number of members in a simplex
    #  nspl     = number of evolution steps for each complex before shuffling
    #  mings    = minimum number of complexes required during the optimization process
    #  maxn     = maximum number of function evaluations allowed during optimization
    #  kstop    = maximum number of evolution loops before convergency
    #  percento = the percentage change allowed in kstop loops before convergency

    # LIST OF LOCAL VARIABLES
    #    x(.,.)    = coordinates of points in the population
    #    xf(.)     = function values of x(.,.)
    #    xx(.)     = coordinates of a single point in x
    #    cx(.,.)   = coordinates of points in a complex
    #    cf(.)     = function values of cx(.,.)
    #    s(.,.)    = coordinates of points in the current simplex
    #    sf(.)     = function values of s(.,.)
    #    bestx(.)  = best point at current shuffling loop
    #    bestf     = function value of bestx(.)
    #    worstx(.) = worst point at current shuffling loop
    #    worstf    = function value of worstx(.)
    #    xnstd(.)  = standard deviation of parameters in the population
    #    gnrng     = normalized geometric mean of parameter ranges
    #    lcs(.)    = indices locating position of s(.,.) in x(.,.)
    #    bound(.)  = bound on ith variable being optimized
    #    ngs1      = number of complexes in current population
    #    ngs2      = number of complexes in last population
    #    iseed1    = current random seed
    #    criter(.) = vector containing the best criterion values of the last
    #                10 shuffling loops

    # Initialize SCE parameters:
    nopt=x0.size
    npg=2*nopt+1
    nps=nopt+1
    nspl=npg
    mings=ngs
    npt=npg*ngs

    bound = bu-bl  #np.array

    # Create an initial population to fill array x(npt,nopt):
    x = SampleInputMatrix(npt,nopt,bu,bl,iseed,distname='randomUniform')
    if iniflg==1:
        x[0,:]=x0

    nloop=0
    icall=0
    xf=np.zeros(npt)
    for i in range (npt):
        xf[i] = EvalObjF(nopt,x[i,:], data=data)
        icall += 1
    f0=xf[0]

    # Sort the population in order of increasing function values;
    idx = np.argsort(xf)
    xf = np.sort(xf)
    x=x[idx,:]

    # Record the best and worst points;
    bestx=x[0,:]
    bestf=xf[0]
    worstx=x[-1,:]
    worstf=xf[-1]

    BESTF=bestf
    BESTX=bestx
    ICALL=icall

    # Compute the standard deviation for each parameter
    xnstd=np.std(x,axis=0)

    # Computes the normalized geometric range of the parameters
    gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

    print('The Initial Loop: 0')
    print(' BESTF:  %f ' %bestf)
    print(' BESTX:  ')
    print(bestx)
    print(' WORSTF:  %f ' %worstf)
    print(' WORSTX: ')
    print(worstx)
    print('     ')

    # Check for convergency;
    if icall >= maxn:
        print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
        print('ON THE MAXIMUM NUMBER OF TRIALS ')
        print(maxn)
        print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
        print(icall)
        print('OF THE INITIAL LOOP!')

    if gnrng < peps:
        print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

    # Begin evolution loops:
    nloop = 0
    criter=[]
    criter_change=1e+5

    while icall<maxn and gnrng>peps and criter_change>pcento:
        nloop+=1

        # Loop on complexes (sub-populations);
        for igs in range(ngs):
            # Partition the population into complexes (sub-populations);
            cx=np.zeros((npg,nopt))
            cf=np.zeros((npg))

            k1=np.array(range(npg))
            k2=k1*ngs+igs
            cx[k1,:] = x[k2,:]
            cf[k1] = xf[k2]

            # Evolve sub-population igs for nspl steps:
            for loop in range(nspl):

                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs=np.array([0]*nps)
                lcs[0] = 1
                for k3 in range(1,nps):
                    for i in range(1000):
                        # lpos = 1 + int(np.floor(npg+0.5-np.sqrt((npg+0.5)**2 - npg*(npg+1)*random.random())))
                        lpos = int(np.floor(npg+0.5-np.sqrt((npg+0.5)**2 - npg*(npg+1)*random.random())))
                        # idx=find(lcs(1:k3-1)==lpos)
                        idx=(lcs[0:k3]==lpos).nonzero()  #check of element al eens gekozen
                        if idx[0].size == 0:
                            break

                    lcs[k3] = lpos
                lcs.sort()

                # Construct the simplex:
                s = np.zeros((nps,nopt))
                s=cx[lcs,:]
                sf = cf[lcs]

                snew,fnew,icall=cceua(s,sf,bl,bu,icall,maxn,iseed,data=data)

                # Replace the worst point in Simplex with the new point:
                s[-1,:] = snew
                sf[-1] = fnew

                # Replace the simplex into the complex;
                cx[lcs,:] = s
                cf[lcs] = sf

                # Sort the complex;
                idx = np.argsort(cf)
                cf = np.sort(cf)
                cx=cx[idx,:]

            # End of Inner Loop for Competitive Evolution of Simplexes
            #end of Evolve sub-population igs for nspl steps:

            # Replace the complex back into the population;
            x[k2,:] = cx[k1,:]
            xf[k2] = cf[k1]

        # End of Loop on Complex Evolution;

        # Shuffled the complexes;
        idx = np.argsort(xf)
        xf = np.sort(xf)
        x=x[idx,:]

        PX=x
        PF=xf

        # Record the best and worst points;
        bestx=x[0,:]
        bestf=xf[0]
        worstx=x[-1,:]
        worstf=xf[-1]

        BESTX = np.append(BESTX,bestx, axis=0) #appenden en op einde reshapen!!
        BESTF = np.append(BESTF,bestf)
        ICALL = np.append(ICALL,icall)

        # Compute the standard deviation for each parameter
        xnstd=np.std(x,axis=0)

        # Computes the normalized geometric range of the parameters
        gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

        print('Evolution Loop: %d  - Trial - %d' %(nloop,icall))
        print(' BESTF:  %f ' %bestf)
        print(' BESTX:  ')
        print(bestx)
        print(' WORSTF:  %f ' %worstf)
        print(' WORSTX: ')
        print(worstx)
        print('     ')

        # Check for convergency;
        if icall >= maxn:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ')
            print(maxn)
            print('HAS BEEN EXCEEDED.')

        if gnrng < peps:
            print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

        criter=np.append(criter,bestf)

        if nloop >= kstop: #nodig zodat minimum zoveel doorlopen worden
            criter_change= np.abs(criter[nloop-1]-criter[nloop-kstop])*100
            criter_change= criter_change/np.mean(np.abs(criter[nloop-kstop:nloop]))
            if criter_change < pcento:
                print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f' %(kstop,pcento))
                print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')

    # End of the Outer Loops
    print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' %icall)
    print('NORMALIZED GEOMETRIC RANGE = %f'  %gnrng)
    print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f' %(kstop,criter_change))

    #reshape BESTX
    BESTX=BESTX.reshape(int(BESTX.size/nopt),int(nopt))

    # END of Subroutine sceua
    return bestx,bestf,BESTX,BESTF,ICALL


def SampleInputMatrix(nrows,npars,bu,bl,iseed,distname='randomUniform'):
    '''
    Create inputparameter matrix for nrows simualtions,
    for npars with bounds ub and lb (np.array from same size)
    distname gives the initial sampling ditribution (currently one for all parameters)

    returns np.array

    '''

    np.random.seed(iseed)
    x=np.zeros((nrows,npars))
    bound = bu-bl
    for i in range(nrows):
        # x[i,:]= bl + DistSelector([0.0,1.0,npars],distname='randomUniform')*bound  #only used in full Vhoeys-framework
        x[i,:]= bl + np.random.rand(1,npars)*bound
        
    return x


def EvalObjF(npar,x,data):
    '''
    Function to calculate the loglikelihood
    '''
    
    # time
    t = data.time.values
    # start
    loglikis = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        # get epsilon, sigma and mu
        epsilon  = x[0]
        # epsilon += x[1]*data.pc1.values[i] + x[2]*data.pc2.values[i] # + \
                   # x[3]*data.pc3.values[i] + x[4]*data.pc4.values[i]
        sigma  = x[1]
        # sigma += x[2]*data.pc1.values[i] + x[3]*data.pc2.values[i]
                 # x[6]*data.pc3.values[i] + x[7]*data.pc4.values[i]
        mu  = x[2]
        mu += x[3]*data.pc1.values[i] + x[4]*data.pc2.values[i] # + \
              # x[7]*data.pc3.values[i] + x[8]*data.pc4.values[i]
        # append for loglike or break
        aux = 1 + (epsilon * ((data.ss.values[i] - mu) / sigma))
        if (aux < 0) or (sigma<0):
            # break model
            return 9999999999999
        else:
            loglikis[i] = math.log(sigma) + (1 + 1/epsilon) * math.log(aux) + aux**(-1/epsilon)
    # calculate loklikelihood
    logLike = sum(loglikis)
    if logLike<0:
        return 9999999999999
    else:
        return logLike
    

def loglikelihood(x, norm, gev):
    lnorm = 0
    lgev = 0
    for i in x:
        lnorm += np.log(stats.norm.pdf(i, *norm))
        lgev += np.log(stats.genextreme.pdf(i, *gev))
    
    return lnorm, lgev


def plot_parameters(BESTX,BESTF,ICALL):
    '''
    plot the trace of the parameters value
    '''

    fig, axes = plt.subplots(ncols=2, figsize=(14,6))
    axes[0].plot(ICALL,BESTX)
    axes[0].set_title('Trace of the different parameters')
    axes[0].set_xlabel('Function calls')
    axes[0].set_ylabel('Parameters values')
    plt.yscale('log')
    axes[1].plot(ICALL,BESTF)
    axes[1].set_title('Trace of the different function values')
    axes[1].set_ylim(0,10000)
    axes[1].set_xlabel('Function calls')
    axes[1].set_ylabel('LogLikelihood')


def plot_adjust(x,data):
    '''
    plot the trace of the parameters and data
    '''
    
    # get epsilon, sigma and mu
    # data.iloc[:,3:] = data.iloc[:,3:].rolling(31).mean()
    epsilon  = np.repeat(x[0], data.shape[0])
    # epsilon += x[1]*data.pc1.values + x[2]*data.pc2.values # + \
               # x[3]*data.pc3.values + x[4]*data.pc4.values
    sigma  = np.repeat(x[1], data.shape[0])
    # sigma += x[2]*data.pc1.values + x[3]*data.pc2.values
             # x[6]*data.pc3.values + x[7]*data.pc4.values
    mu  = x[2]
    mu += x[3]*data.pc1.values + x[4]*data.pc2.values # + \
          # x[7]*data.pc3.values + x[8]*data.pc4.values
    # plot the data and confidence intervals
    fig, axes = plt.subplots(nrows=4, figsize=(14,24))
    for axi, ax in enumerate(axes):
        ax.scatter(data.index[axi*1*365:(axi+1)*1*365], 
                   data.ss[axi*1*365:(axi+1)*1*365], 
                   label='Data', c='k', s=6, 
                   alpha=0.6, zorder=88)
        # ax.plot(data_day.index, epsilon, label='eps')
        # ax.plot(data_day.index, sigma, label='sigma')
        ax.plot(data.index[axi*1*365:(axi+1)*1*365], 
                mu[axi*1*365:(axi+1)*1*365], 
                label='mu', c='seagreen', lw=2, alpha=0.6)
        # p50 = ((((-np.log(0.50))**epsilon)-1)*sigma)/epsilon + mu
        p50 = mu - (sigma/epsilon) * (1 - (-np.log(1-0.5))**(-epsilon))
        # p95 = ((((-np.log(0.05))**epsilon)-1)*sigma)/epsilon + mu
        p95 = mu - (sigma/epsilon) * (1 - (-np.log(1-0.05))**(-epsilon))
        # p05 = ((((-np.log(0.95))**epsilon)-1)*sigma)/epsilon + mu
        p05 = mu - (sigma/epsilon) * (1 - (-np.log(1-0.95))**(-epsilon))
        ax.plot(data.index[axi*1*365:(axi+1)*1*365], 
                p50[axi*1*365:(axi+1)*1*365], 
                '-.', c='crimson', label='P50', alpha=0.75)
        ax.plot(data.index[axi*1*365:(axi+1)*1*365], 
                p95[axi*1*365:(axi+1)*1*365], 
                '--', c='darkred', label='P95', alpha=0.75)
        ax.plot(data[axi*1*365:(axi+1)*1*365].index, 
                p05[axi*1*365:(axi+1)*1*365], 
                '--', c='darkred', label='P05', alpha=0.75)
        ax.legend(loc='upper right')
    fig.suptitle('RESULTS:', fontsize=26)
    # plot the data and PC1
    fig, axes = plt.subplots(nrows=2, figsize=(14,10))
    for axi, ax in enumerate(axes):
        ax.scatter(data.index[axi*1*365:(axi+1)*1*365], 
                   data.ss[axi*1*365:(axi+1)*1*365], 
                   label='Data', c='k', s=6, 
                   alpha=0.9, zorder=88)
        ax.plot(data.index[axi*1*365:(axi+1)*1*365],
                data.pc1.values[axi*1*365:(axi+1)*1*365],
                label='PC1', c='royalblue', alpha=0.7)
    # plot the parameters
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True,
                                        figsize=(14,8))
    plt.subplots_adjust(hspace=0.05)
    ax1.plot(data.index, epsilon, label='eps')
    ax1.set_xticks([])
    ax1.legend(loc='upper right')
    ax2.plot(data.index, sigma, label='sigma')
    ax2.set_xticks([])
    ax2.legend(loc='upper right')
    ax3.plot(data.index, mu, label='mu')
    ax3.legend(loc='upper right')
    fig.suptitle('Value of parameters')
    # plot the return periods
    moment = 250
    # print('The return period is also shown below, calculated with the given parameters: ')
    # print('** Return period in day {} of a normal year **'.format(int(T[moment])))
    fig, ax = plt.subplots(figsize=(14,5))
    plt.xscale('log')
    return_periods = np.arange(1,200,1)
    ssreturn = mu[moment] - (sigma[moment]/epsilon[moment]) * \
        (1 - (-np.log(1-(1/return_periods)))**(-epsilon[moment]))
    ax.plot(return_periods, ssreturn)
    ax.set_xlabel('Return period in years')
    ax.set_ylabel('Storm Surge [m]')
    ax.grid()
    fig.suptitle('Return period in years for SS [m]')
    
