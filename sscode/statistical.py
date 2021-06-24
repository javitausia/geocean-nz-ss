# array
import numpy as np
import pandas as pd
import xarray as xr

# scipy (stats)
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t, chi2 # for loglikelihood ratio test
from scipy.special import ndtri  # norm inv
# genextreme
from pyextremes import EVA
from scipy.stats import genextreme, gumbel_r

# progressbar
import progressbar

# cutom
from .plotting.validation import plot_gev_stats


def gev_matrix(X_set, lon, lat, try_gumbel: bool = True,
               plot: bool = False, verbose: bool = True,
               quantile_th = None, num_tries: int = 8,
               lr_test: bool = False, cluster_number: int = -1,
               gev_title='GEV parameters plot!!'):
    """
    This function is very useful when the GEV analysis wants to
    be done over a spatial region, been this region data saved in 
    an xarray.Dataset format, which is the main dataformat of the project.
    The python package used to calculate the statistical models is
    completly explained at https://github.com/georgebv/pyextremes,
    and it uses scipy.stats functions in the hood

    Args:
        X_set (xarray.Dataset): This is the main xarray dataset, where
            the variable to analyze, which usually is the SS, must be stored,
            but also the longitudes, latitudes and time
        lon (xarray.Coord name): This is the name of the coordinate longitude
        lat (xarray.Coord name): This is the name of the coordinate latitude
        try_gumbel (bool, optional): Wether to try or not to improve the bad results
            using the gumbel distribution. Defaults to True.
        plot (bool, optional): Plot the results if required. Defaults to False.
        verbose (bool, optional): Plot some debugs. Defaults to True.
        quantile_th (float, optional): POT value. This is added as the main python package
            usage implements this by default, but is not used. Defaults to None.
        num_tries (int, optional): Number of times to refit the data in case the
            statistical model is not correctly behaving. Defaults to 8.
        lr_test (bool, optional): Whether to use or not the lokelihood ratio test
            to evaluate model performances
        gev_title (str, optional): Defaults to 'GEV parameters plot!!'.
        cluster_number (int, optional): Defaults to -1.

    Returns:
        [xarray.Dataset]: The statistical models parameters are
            returned by lon / lat
    """

    # TODO: add some debugs

    # pass xarray.Dataset to numpy
    X = X_set.values.reshape(X_set.shape[0],-1)

    # save mu, phi, xi
    mu, phi, xi = [], [], []

    # fit the data
    improves_counter = 0 # check improves

    for i in progressbar.progressbar(range(X.shape[1])):

        ss_series = pd.Series(
            data=X[:,i],name='ss',index=X_set.time.values
        ).dropna() # pass numpy to pandas series for EVA

        if len(ss_series)==0: # input NaNs if no data is available
            mu.append(np.nan), phi.append(np.nan), xi.append(np.nan)
            continue # move to next node

        # create the first model to test
        model = EVA(data=ss_series)
        if quantile_th: # POT with quantile if specified
            model.get_extremes(
                method='POT',threshold=np.nanquantile(X[:,i],quantile_th)
            )
        else: # or all daily/dataset default maxima
            # model.get_extremes(method='POT',threshold=np.nanmin(X[:,i])-0.1)
            model.get_extremes(
                method='BM',extremes_type='high',
                block_size='1D',errors='ignore'
            ) # TODO: add BM method
        model.fit_model(
            distribution=genextreme,model='MLE'
        )
        tries_counter = 0
        p_values_list = []

        # start while loop based on parameter c
        if np.abs(model.distribution.mle_parameters['c'])<0.6:
            mu.append(model.distribution.mle_parameters['loc'])
            phi.append(model.distribution.mle_parameters['scale'])
            xi.append(-model.distribution.mle_parameters['c'])
            continue

        # perform different analysis to get the best possible distribution
        while tries_counter<num_tries:

            # count the times stat-adjustments are tried
            tries_counter += 1

            if try_gumbel and num_tries<2:
                # try new model with gumbel distribution
                new_model = EVA(data=ss_series)
                # new_model.get_extremes(
                #     method='POT',threshold=np.nanmin(X[:,i])-0.1
                # )
                new_model.get_extremes(
                    method='BM',extremes_type='high',
                    block_size='1D',errors='ignore'
                )
                new_model.fit_model(
                    distribution=gumbel_r,model='MLE'
                )
            else:
                # try new model changing the data (not recommended)
                ss_series_drop = ss_series.where(
                    ss_series<ss_series.quantile(0.8)
                ).dropna()
                index_to_delete = np.random.randint(
                    0,len(ss_series_drop),int(len(ss_series_drop)/10)
                )
                new_model = EVA(
                    data=ss_series.drop(
                        ss_series_drop.index[index_to_delete]
                    )
                )
                # new_model.get_extremes(
                #     method='POT',threshold=np.nanmin(X[:,i])-0.1
                # )
                new_model.get_extremes(
                    method='BM',extremes_type='high',
                    block_size='1D',errors='ignore'
                )
                new_model.fit_model(
                    distribution=genextreme,model='MLE'
                )

            # save new statistical model if better   
            if lr_test:
                # check if new model improves the null hypothesis
                l_h = -2 * (
                    new_model.model.loglikelihood - model.model.loglikelihood
                ) # calculate the loglikis difference
                p_value = chi2.sf(l_h,1) # new model has 1 parameter less
                if p_value>0.8:
                    model = new_model # save new model as best
                    p_values_list.append(p_value)
                    mu.append(new_model.distribution.mle_parameters['loc'])
                    phi.append(new_model.distribution.mle_parameters['scale'])
                    try:
                        xi.append(-new_model.distribution.mle_parameters['c'])
                    except:
                        xi.append(0.0) # append the default shape gumbel parameter
                    improves_counter += 1
                    break
                elif tries_counter==num_tries:
                    mu.append(np.nan), phi.append(np.nan), xi.append(np.nan)
                    break
            else:
                # check Akaike criterion
                if new_model.model.AIC<model.model.AIC:
                    model = new_model # save new model as best
                    # save calculated parameters
                    mu.append(new_model.distribution.mle_parameters['loc'])
                    phi.append(new_model.distribution.mle_parameters['scale'])
                    try:    
                        xi.append(-new_model.distribution.mle_parameters['c'])
                    except:
                        xi.append(0.0) # append the default shape gumbel parameter
                    improves_counter += 1
                    break
                elif tries_counter==num_tries:
                    mu.append(np.nan), phi.append(np.nan), xi.append(np.nan)
                    break

    # add improvements print
    print('\n the GEV fit has improved {} times by random/gumbel in cluster {}... \n'.format(
        improves_counter, cluster_number
    )) if verbose else None
    # print('\n the mean of the p-values is {} \n'.format(np.mean(p_values_list)))

    gev_data = X_set.to_dataset(name='ss').assign({
        'mu': ((lon,lat),np.array(mu).reshape(
            len(X_set[lat]),len(X_set[lon])
        ).T),
        'phi': ((lon,lat),np.array(phi).reshape(
            len(X_set[lat]),len(X_set[lon])
        ).T),
        'xi': ((lon,lat),np.array(xi).reshape(
            len(X_set[lat]),len(X_set[lon])
        ).T)
    })

    # plot results
    if plot:
        plot_gev_stats(gev_data,gev_title=gev_title)

    return gev_data
    

def ksdensity_cdf(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x) 
    fint = interp1d(kde.support, kde.cdf)

    return fint(x)


def ksdensity_icdf(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    return fint(p)


def generalizedpareto_cdf(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params 
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf


def generalizedpareto_icdf(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''

    # fit a generalized pareto and get params 
    shape, _, scale = genpareto.fit(x)

    # get percent points (inverse of CDF) 
    icdf = genpareto.ppf(p, shape, scale=scale)

    return icdf


def empirical_cdf(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf


def empirical_icdf(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: revisar que el fill_value funcione correctamente

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)


def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student

    family  - 'gaussian' / 't'  (WARNING: ONLY 'gaussian' implemented)
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student) 

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # TODO: no encaja con los datos. no funciona 
        x = np.linspace(np.min(u), np.max(u),100)
        inv_t = np.ndarray((len(x), u.shape[1]))

        for j in range(u.shape[1]):
            param = t.fit(u[:,j])
            t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
            inv_t[:,j] = t_pdf

        # TODO CORRELACION? NUHAT?
        rhohat = np.corrcoef(inv_n.T)
        nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat


def copularnd(family, rhohat, n):
    '''
    Generates and returns random vectors from a copula

    family  - 'gaussian' / 't'  (WARNING: ONLY 'gaussian' implemented)
    rhohat  - correlation matrix
    n       - number of samples to generate
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u


def copula_simulation(data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    data     - 2D nump.array, each variable in a column
    kernels  - list of kernels for each column at data (KDE | GPareto | ECDF)
    num_sim  - number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_cdf, ksdensity_icdf),
        'GPareto' : (generalizedpareto_cdf, generalizedpareto_icdf),
        'ECDF' : (empirical_cdf, empirical_icdf),
    }

    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernels, ' | '.join(d_kf.keys())
            )
        )

    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(data.shape) * np.nan
    ic = 0
    for d, k in zip(data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula 
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim


def runmean(X, m, modestr):
    '''
    parsed runmean function from original matlab codes.
    '''

    mm = 2*m+1

    if modestr == 'edge':
        xfirst = np.repeat(X[0], m)
        xlast = np.repeat(X[-1], m)
    elif modestr == 'zero':
        xfirst = np.zeros(m)
        xlast = np.zeros(m)
    elif modestr == 'mean':
        xfirst = np.repeat(np.nanmean(X), m)
        xlast = xfirst

    Y = np.concatenate(
        (np.zeros(1), xfirst, X, xlast)
    )
    Y = np.nancumsum(Y)
    Y = np.divide(Y[mm:,]-Y[:-mm], mm)

    return Y

