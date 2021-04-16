#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def normalize(data, ix_scalar, ix_directional, minis=[], maxis=[]):
    '''
    Normalize data subset.      norm = (val - min) / (max - min)

    data            - data to normalize  (numpy.array)
    ix_scalar       - scalar columns indexes  (list[int])
    ix_directional  - directional columns indexes  (list[int])

    minis, maxis    - use to externally set max and min values  (list[float])
    '''

    data_norm = np.zeros(data.shape) * np.nan

    # calculate maxs and mins 
    if minis==[] or maxis==[]:

        # scalar data
        for ix in ix_scalar:
            v = data[:, ix]
            mi = np.amin(v)
            ma = np.amax(v)
            data_norm[:, ix] = (v - mi) / (ma - mi)
            minis.append(mi)
            maxis.append(ma)

        minis = np.array(minis)
        maxis = np.array(maxis)

    # max and mins given
    else:

        # scalar data
        for c, ix in enumerate(ix_scalar):
            v = data[:, ix]
            mi = minis[c]
            ma = maxis[c]
            data_norm[:,ix] = (v - mi) / (ma - mi)

    # directional data (º)
    for ix in ix_directional:
        v = data[:,ix]
        data_norm[:,ix] = v * np.pi / 180.0


    return data_norm, minis, maxis

def denormalize(data_norm, ix_scalar, ix_directional, minis, maxis):
    '''
    Denormalize data subset

    data            - data to denormalize  (numpy.array)
    ix_scalar       - scalar columns indexes  (list[int])
    ix_directional  - directional columns indexes  (list[int])
    minis, maxis    - max and min values  (list[float])
    '''

    data = np.zeros(data_norm.shape) * np.nan

    # scalar data
    for c, ix in enumerate(ix_scalar):
        v = data_norm[:,ix]
        mi = minis[c]
        ma = maxis[c]
        data[:, ix] = v * (ma - mi) + mi

    # directional data
    for ix in ix_directional:
        v = data_norm[:,ix]
        data[:, ix] = v * 180 / np.pi

    return data

def normalized_distance(M, D, ix_scalar, ix_directional):
    '''
    Normalized distance between rows in M and D

    M - numpy array
    D - numpy array
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    dif = np.zeros(M.shape)

    # scalar
    for ix in ix_scalar:
        dif[:,ix] = D[:,ix] - M[:,ix]

    # directional
    for ix in ix_directional:
        ab = np.absolute(D[:,ix] - M[:,ix])
        dif[:,ix] = np.minimum(ab, 2*np.pi - ab)/np.pi

    dist = np.sum(dif**2,1)
    return dist

def nearest_indexes(data_q, data, ix_scalar, ix_directional):
    '''
    for each row in data_q, find nearest point in data and store index.

    Returns array of indexes of each nearest point to all entries in data_q
    '''

    # normalize scalar and directional data 
    data_norm, minis, maxis = normalize(data, ix_scalar, ix_directional)
    data_q_norm, _, _ = normalize(
        data_q, ix_scalar, ix_directional,
        minis=minis, maxis=maxis
    )

    # compute distances, store nearest distance index
    ix_near = np.zeros(data_q_norm.shape[0]).astype(int)
    for c, dq in enumerate(data_q_norm):
        ddq = np.repeat([dq], data_norm.shape[0], axis=0)
        D = normalized_distance(data_norm, ddq, ix_scalar, ix_directional)
        ix_near[c] = np.argmin(D)

    return ix_near

def maxdiss_simplified_no_threshold(data, num_centers, ix_scalar,
                                    ix_directional, log=False):
    '''
    Normalize data and calculate centers using
    maxdiss simplified no-threshold algorithm

    data            - data to apply maxdiss algorithm  (numpy.array)
                      data variables at columns
    num_centers     - number of centers to calculate  (int)
    ix_scalar       - scalar columns indexes  (list[int])
    ix_directional  - directional columns indexes  (list[int])
                      directional data has to be in degrees(º)

    log             - True for printing maxdiss execution log  (bool)
    '''

    # normalize scalar and directional data
    data_norm, minis, maxis = normalize(data, ix_scalar, ix_directional)

    # mda seed
    seed = np.where(data_norm[:,0] == np.amax(data_norm[:,0]))[0][0]

    # initialize centroids subset
    subset = np.array([data_norm[seed]])
    train = np.delete(data_norm, seed, axis=0)

    # repeat till we have desired num_centers
    n_c = 1
    while n_c < num_centers:
        m = np.ones((train.shape[0],1))
        m2 = subset.shape[0]

        if m2 == 1:
            xx2 = np.repeat(subset, train.shape[0], axis=0)
            d_last = normalized_distance(train, xx2, ix_scalar, ix_directional)

        else:
            xx = np.array([subset[-1,:]])
            xx2 = np.repeat(xx, train.shape[0], axis=0)
            d_prev = normalized_distance(train, xx2, ix_scalar, ix_directional)
            d_last = np.minimum(d_prev, d_last)

        qerr, bmu = np.amax(d_last), np.argmax(d_last)

        if not np.isnan(qerr):
            subset = np.append(subset, np.array([train[bmu,:]]), axis=0)
            train = np.delete(train, bmu, axis=0)
            d_last = np.delete(d_last, bmu, axis=0)

            # optional log
            if log:
                fmt = '0{0}d'.format(len(str(num_centers)))
                print('MDA: dataset {3} centroids: {1:{0}}/{2:{0}}'.format(
                    fmt, subset.shape[0], num_centers, data.shape[0]), end='\r')

        n_c = subset.shape[0]
    print('\n')

    # normalize scalar and directional data
    centroids = denormalize(subset, ix_scalar, ix_directional, minis, maxis)

    return centroids

