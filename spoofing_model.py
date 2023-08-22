'''
IMPORTANTE NOTE
---------------
ALL YOU HAVE DONE IS INTENDED FOR AN AGENT THAT WANTS TO BUY A CERTAIN AMOUNT OF SHARES. SHE
TRIES TO SPOOF THE MARKET BY PLACING LIMIT ORDERS AT SOME PRICE LEVEL AT THE ASK SIDE, IN ORDER
TO CREATE A DOWNSIDE PRESSURE AND TAKE ADVANTAGE OF THE DECREASING PRICE. ALL THE ANALYSIS IS 
DONE CONSIDERING THE BEST ASK PRICE AS "TRUE" PRICE.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import argparse
import logging
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn, norm
import numba as nb
import time
import warnings
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import leastsq
import statsmodels.api as sm
import datetime
import sys

def data_preproc(paths_lob, paths_msg, N_days):
    '''This function takes as input the paths of the orderbook and message files for a period of N_days and returns
    the orderbook and message dataframes stacked and with the appropriate columns names and a list containing the
    number of events per day.
    
    Parameters
    ----------
    paths_lob : list
        List of paths of the orderbook files.
    paths_msg : list
        List of paths of the message files.
    N_days : int
        Number of trading days to consider.
    
    Returns
    -------
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    message : pandas dataframe
        Dataframe containing the message data.
    event_per_day : list
        List containing the number of events per day.'''

    if (N_days > len(paths_lob)) or (N_days > len(paths_msg)):
        warnings.warn(f'\nNumber of days considered is greater than the number of days available. \
                       Number of days considered: {N_days}. Number of days available: {len(paths_lob)}.\
                       N_days has been set to {len(paths_lob)}.')
        N_days = len(paths_lob)

    # create the list of dates
    dates = [paths_lob[i].split('/')[-1].split('_')[1] for i in range(N_days)]
    datetimes = []
    for day in dates:
        year, month, day = int(day.split('-')[0]), int(day.split('-')[1].split('0')[1]), int(day.split('-')[2])
        datetimes.append(datetime.datetime(year, month, day))

    orderbook = [pd.read_parquet(f'{paths_lob[i]}') for i in range(N_days)]
    event_per_day = [lob.shape[0] for lob in orderbook]
    message = [pd.read_parquet(f'{paths_msg[i]}') for i in range(N_days)]

    for i in range(N_days):
        message[i][message[i].columns[0]] = message[i][message[i].columns[0]].apply(lambda x: datetimes[i] + timedelta(seconds=x))
        orderbook[i].columns = [f'dummy_column_{i}' for i in range(orderbook[i].shape[1])]
        message[i].columns = [f'dummy_column_{i}' for i in range(message[i].shape[1])]

    message = pd.concat(message, ignore_index=True)
    orderbook = pd.concat(orderbook, ignore_index=True)
    logging.info(f'--------------------\nNumber of trading days considered: \
                  {N_days}\nTotal events: {orderbook.shape[0]}\n------------------------------')
    # Drop the last column of the message dataframe if there are > 6 columns
    if message.shape[1] > 6: message = message.drop(columns=message.columns[-1])

    n = orderbook.shape[1]
    ask_price_columns = [f'Ask price {i}' for i,j in zip(range(1, int(n/2)+1), range(0,n, 4))]
    ask_size_columns = [f'Ask size {i}' for i,j in zip(range(1, int(n/2)+1), range(1,n, 4))]
    bid_price_columns = [f'Bid price {i}' for i,j in zip(range(1, int(n/2)+1), range(2,n, 4))]
    bid_size_columns = [f'Bid size {i}' for i,j in zip(range(1, int(n/2)+1), range(3,n, 4))]
    ask_columns = [[ask_price_columns[i], ask_size_columns[i]] for i in range(len(ask_size_columns))]
    bid_columns = [[bid_price_columns[i], bid_size_columns[i]] for i in range(len(ask_size_columns))]
    columns = np.array([[ask_columns[i], bid_columns[i]] for i in range(len(ask_size_columns))]).flatten()
    orderbook.columns = columns

    message.columns = ['Time', 'Event type', 'Order ID', 'Size', 'Price', 'Direction']

    return orderbook, message, event_per_day

def prices_and_volumes(orderbook):
    '''This function takes as input the orderbook dataframe and returns the bid and ask prices and volumes
    available in the data.
    
    Parameters
    ----------
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    
    Returns
    -------
    bid_prices : numpy array
        Array containing the bid prices.
    bid_volumes : numpy array
        Array containing the bid volumes.
    ask_prices : numpy array
        Array containing the ask prices.
    ask_volumes : numpy array
        Array containing the ask volumes.'''

    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    return bid_prices, bid_volumes, ask_prices, ask_volumes

def volumes_for_imbalance(ask_prices, bid_prices, ask_volumes, bid_volumes, depth):
    '''This function takes as input the bid and ask prices and volumes and returns the volumes at each price level
    for the first depth levels. Note that not all the levels are occupied.
    
    Parameters
    ----------
    ask_prices : numpy array
        Array containing the ask prices.
    bid_prices : numpy array
        Array containing the bid prices.
    ask_volumes : numpy array
        Array containing the ask volumes.
    bid_volumes : numpy array
        Array containing the bid volumes.
    depth : int
        Depth of the LOB.'''

    volume_ask = np.zeros(shape=(ask_prices.shape[0], depth))
    volume_bid = np.zeros(shape=(ask_prices.shape[0], depth))
    for row in range(ask_prices.shape[0]):
        start_ask = ask_prices[row, 0]
        start_bid = bid_prices[row, 0]
        volume_ask[row, 0] = ask_volumes[row, 0] # volume at best
        volume_bid[row, 0] = bid_volumes[row, 0] # volume at best
        for i in range(1,depth): #from the 2th until the 4th tick price level
            if ask_prices[row, i] == start_ask + 100*i: # check if the next occupied level is one tick ahead
                volume_ask[row, i] = ask_volumes[row, i]
            else:
                volume_ask[row, i] = 0

            if bid_prices[row, i] == start_bid - 100*i: # check if the next occupied level is one tick before
                volume_bid[row, i] = bid_volumes[row, i]
            else:
                volume_bid[row, i] = 0
    return volume_ask, volume_bid

def executions_finder(message, f):
    '''This function takes as input the message dataframe and the number of events and returns the executions
    dataframe. It considers all the events after the f-th one.
    
    Parameters
    ----------
    message : pandas dataframe
        Dataframe containing the message data.
    f : int
        Sampling frequency.'''

    message = message.iloc[f:]
    executions = message[message['Event type'] == 4] # Select the market orders
    return executions

def dq_dist(executions, orderbook, depth):
    '''This function takes as input the executions dataframe, the orderbook dataframe
    and the depth and returns a vector containing the price tick deviations due to MOs for each
    MO executed. It considers all the events after the f-th one.
    
    Parameters
    ----------
    executions : pandas dataframe
        Dataframe containing the executions data from the f-th event.
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    depth : int
        Depth of the LOB.
    
    Returns
    -------
    dq : numpy array
        Array containing the price tick deviations due to MOs.'''

    dq = []
    MO_volumes = []
    timestamps = executions['Time'].value_counts().index    # Select the timestamps of executions (i.e. MOs)
    counts = executions['Time'].value_counts().values       # Select the number of executions for each timestamp
    data = {'Time': timestamps, 'Count': counts}            # Create a dictionary with the timestamps as keys and the counts as values
    df = pd.DataFrame(data=data)                            # Create a dataframe with the number of executions for each timestamp (i.e. for each MO)

    for i in range(df.shape[0]):
        executions_slice = executions[executions['Time'] == df['Time'][i]]       # Select all the executions for the specific timestamp. 
                                                                                 # Remember that executions is the message file filtered by event type = 4
        total_volume = executions_slice['Size'].sum()                            # Compute the total volume for of the market order
        MO_volumes.append(total_volume)

        if executions_slice.index[0] - 1 < 0:
            start = 0
        else:
            start = executions_slice.index[0] - 1

        # Each line of the orderbook contains information about the LOB already updated
        # by the action of the corresponding line in the message file. In other words,
        # each line of the orderbook contains the LOB after the action of the corresponding
        # line in the message file. Hence, I have to select the orderbook slice corresponding
        # to the market order. Specifically, I select the rows of the orderbook dataframe that are
        # between the index just before the executions of the market order and the index of its end.
        orderbook_slice = orderbook.iloc[start: executions_slice.index[-1]]  # Select the orderbook slice corresponding to the market order.
                                                                                 # Specifically, I select the rows of the orderbook dataframe that are
                                                                                 # between the index just before the executions of the market order and the index of its end.

        vol, j = 0, 0
        if executions[executions['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1: # If the direction is 1, the price has decreased and I have to consider the bid side
            while vol <= total_volume and j < depth:
                j += 1
                vol += orderbook_slice[f'Bid size {j}'].iloc[0]
            # If the volume of the first j-1 levels is just the same as the total volume of the MO,
            # such MO has completely depleted the first j-1 levels of the LOB. Hence, the MO has
            # caused a price deviation equal to j-1 ticks. other wise, the j-1th level has not been
            # completely depleted and the MO has caused a price deviation equal to j-2 ticks.
            if vol == total_volume:
                tick_shift = int((orderbook_slice[f'Bid price {j+1}'].iloc[0] - orderbook_slice[f'Bid price {1}'].iloc[0])/tick)
            else:
                if j == 1:
                    tick_shift = 0
                else:
                # tick_shift = -(j - 2)
                    tick_shift = int((orderbook_slice[f'Bid price {j}'].iloc[0] - orderbook_slice[f'Bid price {1}'].iloc[0])/tick)
        else:                                                                         # If the direction is -1, the price has increased and I have to consider the ask side
            while vol <= total_volume and j < depth:
                j += 1
                vol += orderbook_slice[f'Ask size {j}'].iloc[0]
            if vol == total_volume:
                # tick_shift = (j - 1)
                tick_shift = int((orderbook_slice[f'Ask price {j+1}'].iloc[0] - orderbook_slice[f'Ask price {1}'].iloc[0])/tick)
            else:
                if j == 1:
                    tick_shift = 0
                else:
                # tick_shift = (j - 2)
                    tick_shift = int((orderbook_slice[f'Ask price {j}'].iloc[0] - orderbook_slice[f'Ask price {1}'].iloc[0])/tick)

        if np.abs(tick_shift) > depth:                                                # If the number of ticks is greater than the depth, discard that value
            pass
        else:
            dq.append(tick_shift)

    
    return dq

def MO_volumes(message, f):
    '''This function takes as input the message dataframe and the sampling frequency and returns the volumes
    of the market orders.
    
    Parameters
    ----------
    message : pandas dataframe 
        Dataframe containing the message data.
    f : int
        Sampling frequency.
        
    Returns
    -------
    MO_volumes : numpy array
        Array containing the volumes of the market orders.'''

    executions = executions_finder(message, f) # Select the executions from the f-th event
    executions = executions[executions['Direction'] == -1] # Select the market orders initiated by a sell order.
                                                           # This means that an agent has bought causing the price 
                                                           # to increase.
    H = np.zeros(message.shape[0]) # H will be the market order volume at each timestamp of
                    # a market order execution. It will be something  that is
                    # zero when there is no market order and then a decreasing
                    # function when there is a market order.
    t = executions['Time'].value_counts().index # Select the timestamps relative to a market order
    c = executions['Time'].value_counts().values # Select the number of executions for each market order
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)
    # idxs contains all the indexes of the message files that are 1-event before every block
    # of market orders. For instance, if there are 3 market orders at time 1, 2 and 3, idxs will
    # contain 0.
    # [executions['Time'] == time] - > Select all the executions at time t=time drawn from df
    # .index.min() - > Select the minimum index of the market order block at time t=time
    # -1 -> Select the index that is 1-event before the first market order of the block
    idxs_m = np.array([(executions[executions['Time'] == time].index.min() - 1) for time in df['Time']]) # Select all the indexes that are 1-event before the 
                                                                                                       # first market order of every blocks (of MOs)
    idxs_p = np.array([(executions[executions['Time'] == time].index.max()) for time in df['Time']])
    idxs_m.sort()
    idxs_p.sort()
                                                                                        
    for i in tqdm(range(df.shape[0])):
        # market_order = executions.iloc[executions['Time'] == df['Time'][i]] # Select the market order from the message file
        market_order = executions.loc[executions['Time'] == df['Time'][i]]

        volumes = market_order['Size']
        # volumes.index contains the indexes of the message file that correspond to the market order
        for j, k in zip(volumes.index, range(volumes.shape[0])): # Compute the volume of the market order at each timestamp
            H[j] = volumes.iloc[k]

    return H, idxs_m, idxs_p

def lob_reconstruction(N, tick, bid_prices, bid_volumes, ask_prices, ask_volumes):
    '''This function takes as input the number of events, the tick size, the minimum and maximum values of the prices,
    the bid and ask prices and volumes and returns the limit order book snapshots for each event.

    Parameters
    ----------
    N : int
        Number of events.
    tick : float
        Tick size.
    bid_prices : pandas dataframe
        Dataframe containing the bid prices.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_prices : pandas dataframe
        Dataframe containing the ask prices.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    
    Returns
    -------
    lob_snapshots : list
        List of the limit order book snapshots for each event.'''

    n_columns = bid_prices.shape[1]
    m, M = bid_prices.min().min(), ask_prices.max().max()
    for event in tqdm(range(N), desc='Computing LOB snapshots'):
        # Create the price and volume arrays
        p_line = np.arange(m, M+tick, tick)
        volumes = np.zeros_like(p_line)

        # Create two dictionaries to store the bid and ask prices keys and volumes as values
        d_ask = {ask_prices[event][i]: ask_volumes[event][i] for i in range(int(n_columns))}
        d_bid = {bid_prices[event][i]: bid_volumes[event][i] for i in range(int(n_columns))}
        mid_price = bid_prices[event][0] + 0.5*(ask_prices[event][0] - bid_prices[event][0])

        # Create two boolean arrays to select the prices in the p_line array that are also in the bid and ask prices
        mask_bid, mask_ask = np.in1d(p_line, list(d_bid.keys())), np.in1d(p_line, list(d_ask.keys()))

        # Assign to the volumes array the volumes corresponding to the the bid and ask prices
        volumes[np.where(mask_bid)] = list(d_bid.values())
        volumes[np.where(mask_ask)] = list(d_ask.values())

        # Insert the mid price in the p_line array and the corresponding volume in the volumes array
        max_bid_loc = np.array(np.where(mask_bid)).max()
        min_bid_loc = np.array(np.where(mask_bid)).min()
        min_ask_loc = np.array(np.where(mask_ask)).min()
        max_ask_loc = np.array(np.where(mask_ask)).max()
        mid_price_loc = max_bid_loc + int(0.5 * (min_ask_loc - max_bid_loc))
        p_line = np.insert(p_line, mid_price_loc, mid_price)
        volumes = np.insert(volumes, np.array(np.where(mask_ask)).min(), 0)

        # Create the colors array to color the bars of the plot
        bid_color = ['g' for i in range(p_line[:mid_price_loc].shape[0])]
        ask_color = ['r' for i in range(p_line[mid_price_loc:].shape[0])]
        colors = np.hstack((bid_color, ask_color))

        # Create the tick positions and labels
        X = np.zeros_like(p_line)
        X[np.where(mask_bid)] = list(d_bid.keys())
        X[np.where(mask_ask)[0] + 1] = list(d_ask.keys())
        tick_positions = np.nonzero(X)[0]
        tick_labels = p_line[tick_positions].astype(int)

        # Plot the limit order book snapshot
        plt.figure(tight_layout=True, figsize=(13,5))
        plt.title(f'Limit Order Book {event}')
        plt.bar(np.arange(p_line.shape[0]), volumes, width=1, color=colors)
        plt.vlines(mid_price_loc, 0, volumes.max(), color='black', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.xlim([min_bid_loc - 10, max_ask_loc + 10])
        plt.savefig(f'lob_snapshots/lob_snapshot_{event}_{info[0]}.jpg')
        plt.close()

def lob_video(image_folder, info):
    '''This function takes as input the path of the folder containing the limit order book snapshots and returns
    the limit order book video.

    Parameters
    ----------
    image_folder : str
        Path of the folder containing the limit order book snapshots.
    
    info: str
        String containing information about the stock and the period of time.

    Returns
    -------
    None.'''

    frame_width = 1300
    frame_height = 500
    fps = 24.0
    output_filename = f"LOBvideo_{info[0]}_{info[1]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    image_files = sorted(os.listdir(image_folder))  # Sort files in ascending order

    img = []
    for image_file in tqdm(image_files, desc='Creating LOB video'):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (frame_width, frame_height))
        video_writer.write(image)

    cv2.destroyAllWindows()
    video_writer.release()

def time_interval_ret_dist(ask_price, tick):
    ask_price_tick = ask_price / tick
    ask_price_diff = np.diff(ask_price_tick)
    mask = ask_price_diff != 0
    diff_idxs = np.where(mask)
    idxs_diff = np.diff(diff_idxs)[0]
    return int(np.mean(idxs_diff))

@nb.njit
def dot_product(a, b):
    '''This function takes as input two numpy arrays and returns their dot product.
    
    Parameters
    ----------
    a : numpy array
        First array.
    b : numpy array
        Second array.
    
    Returns
    -------
    result : float
        Dot product of the two arrays.'''

    result = 0.0
    if len(a) != len(b):
        raise ValueError(f'Shapes not aligned: {len(a)} != {len(b)}')
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def volumes_fsummed(f, depth, bid_volumes, ask_volumes):
    '''This function takes as input the sampling frequency, the depth, the bid and ask volumes and returns
    the bid and ask volumes summed over f events.
    
    Parameters
    ----------
    f : int
        Sampling frequency.
    depth : int
        Depth of the LOB (max price level for ask and min price level for bid)
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    
    Returns
    -------
    bid_volumes_fsummed : numpy array
        Array containing the bid volumes summed over f events.
    ask_volumes_fsummed : numpy array
        Array containing the ask volumes summed over f events.'''

    bid_volumes, ask_volumes = bid_volumes[:, :depth], ask_volumes[:, :depth]
    bid_volumes_fsummed = np.array([bid_volumes[i:i+f].sum(axis=0) for i in range(0, bid_volumes.shape[0]-f)])
    ask_volumes_fsummed = np.array([ask_volumes[i:i+f].sum(axis=0) for i in range(0, bid_volumes.shape[0]-f)])

    return bid_volumes_fsummed, ask_volumes_fsummed

# @nb.njit
def avg_imbalance_faster(bid_volumes_fsummed, ask_volumes_fsummed, weight, f):
    '''This function is a faster version of avg_imbalance. The speed up is made possible by the numba library.
    In the original function, some non supported numba functions are used. This version is not able to compute
    the errors on the estimated parameters and is not able to save the output. It is mainly used during the 
    computation of the distribution dp+ and the weights w.'''

    imb = []
    for i in range(0, bid_volumes_fsummed.shape[0] - f):

        num = dot_product(bid_volumes_fsummed[i], weight)
        det = dot_product((bid_volumes_fsummed[i] + ask_volumes_fsummed[i]), weight)
        imb.append(num/det)

    imb = np.array(imb)
    return imb

def avg_imbalance(bid_volumes, ask_volumes, weight, f, bootstrap=False, save_output=False):
    '''This function takes as input, the bid and ask volumes, the weight and the sampling frequency
    and returns the average imbalance (over f events).

    Parameters
    ----------
    num_events : int
        Number of events.
    bid_volumes : numpy array
        Array containing the bid volumes.
    ask_volumes : numpy array
        Array containing the ask volumes.
    weight : numpy array
        Array containing the weight.
    f : int
        Frequency.
    bootstrap : Bool
        If True compute the errors via bootstrap. The default is False.
    save_output : Bool
        If True save the output. The default is False.
    Returns
    -------
    imb : numpy array
        Array containing the average imbalance.'''

    lev = weight.shape[0]
    imb = []
    for i in tqdm(range(0, bid_volumes.shape[0] - f), desc='Computing average imbalance'):
        num = dot_product(bid_volumes[i:i+f, :lev].sum(axis=0), weight)
        det = dot_product((bid_volumes[i:i+f, :lev].sum(axis=0) + ask_volumes[i:i+f, :lev].sum(axis=0)), weight)
        imb.append(num/det)
    imb = np.array(imb)
    shape, loc, scale = stats.skewnorm.fit(imb)
    
    if bootstrap == True:
        n_iterations = 100  # Number of bootstrap iterations
        n_samples = len(imb)  # Number of samples in your data

        # Initialize arrays to store parameter values from each iteration
        shape_samples = np.zeros(n_iterations)
        loc_samples = np.zeros(n_iterations)
        scale_samples = np.zeros(n_iterations)

        # Perform bootstrapping and refit the skew normal distribution in each iteration
        for i in tqdm(range(n_iterations), desc='Computing standard errors'):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = imb[bootstrap_indices]
            shape_samples[i], loc_samples[i], scale_samples[i] = stats.skewnorm.fit(bootstrap_sample)

        # Calculate the standard errors of the parameters
        shape_error = np.std(shape_samples)
        loc_error = np.std(loc_samples)
        scale_error = np.std(scale_samples)

        if save_output:
            np.save(f'output/imbalance_{info[0]}_{info[1]}_errors', np.array([shape_error, loc_error, scale_error]))
    
    else:
        if save_output:
            np.save(f'output/imbalance_{info[0]}_{info[1]}', imb)
            np.save(f'output/imbalance_{info[0]}_{info[1]}_params', np.array([shape, loc, scale]))
        shape_error, loc_error, scale_error = 0, 0, 0

    return imb, np.array([shape, loc, scale]), np.array([shape_error, loc_error, scale_error])

def MO_imbalances(message, imbalance, f):
    '''This function takes as input the message dataframe, the imbalance and the sampling frequency and returns
    the imbalances before and after each MO.
    
    Parameters
    ----------
    message : pandas dataframe
        Dataframe containing the message data.
    imbalance : numpy array
        Array containing the imbalance.
    f : int
        Sampling frequency.
    
    Returns
    -------
    i_mm : numpy array
        Array containing the imbalance just before a sell MO.
    i_pm : numpy array
        Array containing the imbalance just after a sell MO.
    i_mp : numpy array
        Array containing the imbalance just before a buy MO.
    i_pp : numpy array
        Array containing the imbalance just after a buy MO.'''

    message = message.iloc[f:N].reset_index(drop=True) # the first element of imbalance is the f-th one.
                                                       # Since I have to take the values of the imbalance just before
                                                       # and after a market order, I have to consider all the market orders
                                                       # executed after the f-th event. With reset_index I reset the index
                                                       # of the message dataframe so that it coincides with the index of
                                                       # imbalance.
    executions = message[message['Event type'] == 4] # Here I actually select the market orders after the f-th event
    executions_sell = executions[executions['Direction'] == -1] # Select the market orders initiated by a sell order.
    executions_buy = executions[executions['Direction'] == 1] # Select the market orders initiated by a buy order.

    t = executions_sell['Time'].value_counts().index
    c = executions_sell['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df_sell = pd.DataFrame(data=d) # Create a dataframe with the number of executions for each timestamp

    t = executions_buy['Time'].value_counts().index
    c = executions_buy['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df_buy = pd.DataFrame(data=d) # Create a dataframe with the number of executions for each timestamp
  
    # Create two arrays containing the imbalance just before and after a market order.
    # For istance, i_m is evaluated considering the last imbalance value just before the
    # beginning of the market order. The index of this last imbalance value is found by
    # selecting the minimum index of the executions dataframe for each timestamp and taking
    # the previous index value. A similar (but specular) procedure is applied to i_p, but in
    # this case you do not take the +1 index.
    i_mm = np.array([imbalance[np.array(executions_sell[executions_sell['Time'] == time].index).min() - 1] for time in tqdm(df_sell['Time'], desc='Computing i_m_sell')])
    i_pm = np.array([imbalance[np.array(executions_sell[executions_sell['Time'] == time].index).max()] for time in tqdm(df_sell['Time'], desc='Computing i_p_sell')])

    i_mp = np.array([imbalance[np.array(executions_buy[executions_buy['Time'] == time].index).min() - 1] for time in tqdm(df_buy['Time'], desc='Computing i_m_buy')])
    i_pp = np.array([imbalance[np.array(executions_buy[executions_buy['Time'] == time].index).max()] for time in tqdm(df_buy['Time'], desc='Computing i_p_buy')])

    np.save(f'output/i_m_sell_{info[0]}_{info[1]}_{f}_{N}', i_mm)
    np.save(f'output/i_p_sell_{info[0]}_{info[1]}_{f}_{N}', i_pm)
    np.save(f'output/i_m_buy_{info[0]}_{info[1]}_{f}_{N}', i_mp)
    np.save(f'output/i_p_buy_{info[0]}_{info[1]}_{f}_{N}', i_pp)

    return i_mm, i_pm, i_mp, i_pp

def square_dist_ispoof(i, i_m_sell, w_k, Q_k, rho_t, mu_p, v_k):
    '''This function compute the implicit equation used to compute numerically the i_spoof value.
    
    Parameters
    ----------
    i : float
        Independent variable.
    i_m_sell : float
        Imbalance just before a MO.
    w_k : float
        Weight k.
    Q_k : float
        Cumulative probability of a price change due to MOs greater than k ticks.
    rho_t : float
        Ratio between the volume of the MO and the volume of the LOB just before the MO.
    mu_p : float
        Average price change due to MOs.
    v_k : float
        Average price change due to MOs greater than k ticks.
    
    Returns
    -------
    (lhs - rhs)**2 : float
        Square of the difference between the left hand side and the right hand side of the equation.'''

    lhs = 1 / i
    rhs = np.array([1 / i_m_sell + (1 - i_m_sell) / i_m_sell * w_k / Q_k * \
                            max(0,(2 * rho_t * w_k * mu_p * (1 - i_m_sell) / i_m_sell * i[t]**2 -  (Q_k * rho_t + v_k))) for t in range(i.shape[0])])
    return (lhs - rhs)**2

def i_spoofing(dq, dp_p, H, weight, depth, f, volume_ask_imb, volume_bid_imb, idxs_m, idxs_p, x0):
    '''This function takes as input the price tick deviations due to MOs, the probability distribution of the price
    changes due to LOs, the market order volumes, the weight, the depth, the sampling frequency, the volumes of the
    LOB, the indexes of the MOs, the initial guess for the i_spoof value and returns the i_spoof
    value.
    
    Parameters
    ----------
    dq : numpy array
        Array containing the distribution od the price tick deviations due to MOs.
    dp_p : numpy array
        Array containing the distribution of the price tick deviations due to LOs.
    H : numpy array
        Array containing the market order volumes.
    weight : numpy array
        Array containing the weight.
    depth : int
        Depth of the LOB.
    f : int
        Sampling frequency.
    volume_ask_imb : numpy array
        Array containing the ask volumes of the LOB.
    volume_bid_imb : numpy array
        Array containing the bid volumes of the LOB.
    idxs_m : numpy array
        Array containing the indexes of the sell MOs.
    idxs_p : numpy array
        Array containing the indexes of the buy MOs.
    x0 : float
        Initial guess for the i_spoof value.
    
    Returns
    -------
    i_spoof : numpy array
        Array containing the i_spoof values.
    v_spoof : numpy array
        Array containing the v_spoof values.''' 

    # These vectors have the same number of elements as the number of MOs
    rho = np.zeros(idxs_m.shape[0])
    a_m = np.zeros(idxs_m.shape[0])
    b_m = np.zeros(idxs_m.shape[0])
    a_p = np.zeros(idxs_p.shape[0])
    b_p = np.zeros(idxs_p.shape[0])

    for t, tt in zip(idxs_m, np.arange(idxs_m.shape[0])):
        a_m[tt] = dot_product(volume_ask_imb[t:t-f:-1][::-1].sum(axis=0), weight) / (weight.shape[0]*f)
        b_m[tt] = dot_product(volume_bid_imb[t:t-f:-1][::-1].sum(axis=0), weight) / (weight.shape[0]*f)
        if np.isnan(a_m[tt]) or np.isnan(b_m[tt]) or np.isinf(a_m[tt]) or np.isinf(b_m[tt]) or a_m[tt] == 0 or b_m[tt] == 0 or volume_ask_imb[t:t+f].shape[0] == 0:
            print(f'\nt={t}, tt={tt}')
            print(f'\nvolume_ask_imb[t:t-f:-1].sum(axis=0)={volume_ask_imb[t:t-f:-1][::-1].sum(axis=0)}')
            print(f'\nvolume_bid_imb[t:t-f:-1].sum(axis=0)={volume_bid_imb[t:t-f:-1][::-1].sum(axis=0)}')
            print(f'\nweight={weight}')
            print(f'\nweight.shape[0]={weight.shape[0]}')
            print(f'\nf={f}')
        rho[tt] = H[t:t-f:-1][::-1].sum() / a_m[tt]
    
    for t, tt in zip(idxs_p, np.arange(idxs_p.shape[0])):
        a_p[tt] = dot_product(volume_ask_imb[t:t-f:-1][::-1].sum(axis=0), weight) / (weight.shape[0]*f)
        b_p[tt] = dot_product(volume_bid_imb[t:t-f:-1][::-1].sum(axis=0), weight) / (weight.shape[0]*f)

    i_initial_m = b_m / (a_m + b_m)
    i_initial_p = b_p / (a_p + b_p)

    mu_p = sum(value * probability for value, probability in zip(np.arange(-depth, depth+1), dp_p))
    
    i_spoof_k = np.zeros(shape=[i_initial_m.shape[0], depth]) # One value of i_spoof for each time and each level k
    v_spoof = np.zeros(shape=[i_initial_m.shape[0], depth])
    Q = np.zeros(shape=depth)
    v = np.zeros(shape=depth)
    for k in tqdm(range(depth), desc='Levels'):
        Q[k] = dq[depth + k:].sum()
        v[k] = np.sum([dq[depth+k:][i]*(i) for i in range(len(dq[depth+k:]))])
        # v[k] = np.array([(i-k) * dq[depth + k:] for i in range(depth + k, 2*depth+1)]).sum()
        for t in tqdm(range(i_initial_m.shape[0])):
            solution = leastsq(square_dist_ispoof, x0=x0, args=(i_initial_m[t], weight[k], Q[k], rho[t], mu_p, v[k]))
            i_spoof_k[t, k] = solution[0]
            v_spoof[t, k] = a_m[t] / Q[k] * max(0, (2 * rho[t] * weight[k] * mu_p * (1 - i_initial_m[t]) / i_initial_m[t] * i_spoof_k[t, k]**2 - (Q[k] * rho[t] + v[k])))

    spoof_term = np.zeros_like(v_spoof)
    for t in range(spoof_term.shape[0]):
        for l in range(weights.shape[0]):
            spoof_term[t,l] = v_spoof[t,l] * weights[l]
    spoof_term = spoof_term.sum(axis=1)

    i_spoof = b_m / (a_m + b_m + spoof_term)

    return i_spoof_k, v_spoof, i_spoof, i_initial_m, i_initial_p, [Q, v, a_m, b_m, rho, mu_p]

def partition_into_quantiles(v, N):
    '''This function takes as input an array and the number of quantiles and returns the quantiles.

    Parameters
    ----------
    v : numpy array
        Array containing the values.
    N : int
        Number of quantiles.
    
    Returns
    -------
    quantiles : numpy array
        Array containing the quantiles.'''

    sorted_v = np.sort(v)
    inc = (len(v) - 1) / N
    breakpoints = [round(i * inc) for i in range(1, N)]
    
    quantiles = []
    quantiles.append(sorted_v[:breakpoints[0]+1])
    for i in range(1, N-1):
        quantiles.append(sorted_v[breakpoints[i-1]+1:breakpoints[i]+1])
    quantiles.append(sorted_v[breakpoints[N-2]+1:])
    quantiles = np.array(quantiles, dtype=object)
    for i in range(quantiles.shape[0]):
        quantiles[i] = np.resize(quantiles[i], quantiles[3].shape[0])
    return quantiles

def dp_distribution_objective(parameters, M, ask_prices_fdiffs, bid_volumes_fsummed, ask_volumes_fsummed, frequencies, depth):
    '''This function takes as input the parameters, the number of levels, the bid and ask volumes,
    the sampling frequency and returns the (negative) objective function that has to be minimize to 
    estimate the values of dp^+ and w (ML estimation).
    
    Parameters
    ----------
    parameters : numpy array
        Array containing the parameters.
    M : int
        Number of levels.
    num_events : int
        Number of events.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    f : int
        Frequency.
        
    Returns
    -------
    obj_fun : float
        Objective function that has to be minimized.'''

    # set the first M values for dp (depth*2 + 1)
    dp = parameters[:M]
    # set the last M values for w (depth + 1)
    w = parameters[M:]
    obj_fun = 0
    D = 0

    # Evalute the imbalance for each day in N days using the weights w
    # of the current iteration of the optimization algorithm.
    imbalance = []
    for day in range(len(ask_prices_fdiffs)):
        f = frequencies[day]
        # Compute the imbalance for each days in N days
        l = avg_imbalance_faster(bid_volumes_fsummed[day], ask_volumes_fsummed[day], w, f)
        imbalance.append(l)
        D += l.shape[0]

    # Compute the parameters of the a skew normal distribution fitted
    # on the overall imbalance.
    shape, loc, scale = stats.skewnorm.fit(np.concatenate(imbalance))

    # Now for every day compute the objective function
    for day in range(len(ask_prices_fdiffs)):
        # Sample the imbalance every f events. Note that the imbalance is
        # computed averaging the f previous events every time. It starts
        # from the f-th event and goes on until the end of the day.
        # In order to have the pair (x_m, i_m) where x_m represents the price
        # change every f events "influenced" by i_m (so I see i_m at time t and 
        # check the price change at time t+f), I have to sample the imbalance
        # every f events.
        f = frequencies[day]
        imb = imbalance[day][::f]
        for i in range(imb.shape[0] - 1):
            # If the difference exceeds the max depth, ignore it
            if np.abs(ask_prices_fdiffs[day][i]) > depth:
                pass
            else:
                dp_pp = dp[ask_prices_fdiffs[day][i] + depth]
                dp_pm = dp[-ask_prices_fdiffs[day][i] + depth]
                p_im = stats.skewnorm.pdf(imb[i], shape, loc, scale)

                # Compute the objective function (negative log-likelihood)
                obj_fun += - np.log((imb[i] * dp_pp + (1 - imb[i]) * dp_pm) * p_im)

    return obj_fun/D

def moments(values, probabilities):
    '''This function computes the first three moments of a discrete distribution.'''
    mu_p = sum([v*p for v,p in zip(values, probabilities)])
    sgm = np.sqrt(sum([(v-mu_p)**2*p for v,p in zip(values, probabilities)]))
    skew = sum([(v-mu_p)**3*p for v,p in zip(values, probabilities)]) / sgm**3
    return mu_p, sgm, skew

def callback(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, f, iteration):
    print(f"Iteration: {iteration}")
    with open(f'opt_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write(f"\nStock: {info[0]}")
        file.write(f"\nIteration: {iteration}")
        file.write(f"\nCurrent solution for dp:\n{x[:M]} -> sum = {x[:M].sum()}")
        file.write(f"\nCurrent solution for w:\n{x[M:]} -> sum = {x[M:].sum()}")
        file.write(f"\nObjective value:\n{dp_distribution_objective(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, f, depth)}")
        file.write(f"\nMean, Std, Skewness:\n{moments(np.arange(-depth, depth+1), x[:M])}")
        file.write("-------------------------------\n")
    
    if iteration > 0:
        plt.figure(tight_layout=True, figsize=(7,5))
        plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), x[:M])
        plt.savefig(f'images/dp_p_{iteration}_{info[0]}_{info[1]}_{job_number}.png')
        plt.close()
    obj_fun.append(dp_distribution_objective(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, f, depth))

def create_callback(M, ask_prices_fdiff, bid_volumes, ask_volumes, f, depth):
    iteration = 0
    def callback_closure(x, *args):
        if len(args) == 1:
            state = args[0]
        nonlocal iteration
        iteration += 1
        callback(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, f, iteration)
    return callback_closure

def constraint_fun1(x, depth):
    return x[:x.shape[0] - depth].sum() - 1

def constraint_fun2(x, depth):
    return x[x.shape[0] - depth:].sum() - 1

def create_bar_plot(array1, array2, x_labels, labels, title):
    x = np.arange(len(x_labels))  # Generate x-axis values

    width = 0.35  # Width of each bar

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, array1, width, label=labels[0])
    bars2 = ax.bar(x + width/2, array2, width, label=labels[1])

    ax.set_xlabel('Depth')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.legend()
    # Save the image
    plt.savefig(f'images/dp_empvstheo_{info[0]}_{info[1]}_{title}_{job_number}.png')
    plt.close()

def bivariate_skew_normal_pdf(X, mu, sigma, alpha):
    # X: Array-like, shape (2,)
    # mu: Array-like, shape (2,)
    # sigma: Array-like, shape (2,)
    # rho: Correlation coefficient
    # alpha: Array-like, shape (2,)
    #array([-2.15365618e-01, -4.86435919e-01,  4.70898052e-01,  1.96679660e-01,
        # 2.37943107e-01,  1.20902653e+02,  2.32282096e+02])
    
    mean = np.array([0-mu[0], 0-mu[1]])
    # Calculate the covariance matrix
    cov_matrix = np.array([[sigma[0], sigma[1]],
                           [sigma[1], sigma[2]]])
    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
    if min_eig < 0:
        cov_matrix -= 10*min_eig * np.eye(*cov_matrix.shape)
    norm_pdf = mvn.pdf(X, mean, cov_matrix, allow_singular=True)
    norm_cdf = mvn.cdf(alpha*X, mean*alpha, cov_matrix, allow_singular=True)
    pdf = 2 * norm_pdf * norm_cdf
    # Calculate the standardized variables
    # # Calculate the PDF of the multivariate normal distribution
    # normal_pdf = mvn.pdf(Z, mean=mu, cov=cov_matrix, allow_singular=True)
    # # Calculate the CDF of the standard normal distribution
    # cdf_1 = norm.cdf(alpha[0]*X[0]/np.sqrt(1+alpha[0]**2))
    # cdf_2 = norm.cdf(alpha[1]*X[1]/np.sqrt(1+alpha[1]**2))
    # # Calculate the PDF of the bivariate skew normal distribution
    # pdf = 2 * normal_pdf * cdf_1 * cdf_2
    
    return pdf

def neg_log_likelihood_BSKN(params, data):
    mu = params[:2]
    sigma = params[2:5]
    alpha = params[5:7]
    
    # Calculate the PDF values for the data points
    pdf_values = bivariate_skew_normal_pdf(data, mu, sigma, alpha)
    
    # Calculate the negative log-likelihood
    neg_log_likelihood = -np.sum(np.log(pdf_values))
    
    return neg_log_likelihood

def multivariate_skewnorm(params, data):
    mean = params[:2]
    a = params[5:]
    cov_matrix = np.array([[params[2],params[3]], [params[3],params[4]]])
    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))

    if min_eig < 0:
        cov_matrix -= 10*min_eig * np.eye(*cov_matrix.shape)

    dim = len(a)
    data = _process_quantiles(data, dim)
    pdf_val = mvn(mean, cov_matrix).logpdf(data)
    cdf_val = norm(0, 1).logcdf(np.dot(data, a))
    return -np.sum(np.log(2) + pdf_val + cdf_val)/data.shape[0]

    #return logpdf

def _process_quantiles(x, dim):
    if x.ndim == 1:
        x = x.reshape(1, dim)
    return x

def conditional_joint_skn_pdf(y, mu, cov_matrix, alpha, num_samples):
    x = np.linspace(0, 1, num_samples)
    xi_1 = mu[0] + cov_matrix[0,1]/cov_matrix[1,1] * (y - mu[1])
    if np.isnan(xi_1):
        print(f'\ni_1={xi_1}')
    w112 = cov_matrix[0,0] - cov_matrix[0,1]**2/cov_matrix[1,1]
    print(f'\nw112={w112}')
    if np.isnan(w112) or np.isinf(w112) or w112 == 0:
        print(f'\nw112={w112}')
    alpha_2 = (alpha[1] + np.sqrt(cov_matrix[0,1]**2/(cov_matrix[0,0]*cov_matrix[1,1])) * alpha[0]) / (np.sqrt(1 + w112/cov_matrix[0,0] * alpha[0]**2))
    print(f'\nalpha_2={alpha_2}')
    if np.isnan(alpha_2) or np.isinf(alpha_2) or alpha_2 == 0:
        print(f'\nalpha_2={alpha_2}')
    x0 = alpha_2 / np.sqrt(cov_matrix[1,1]) * (y - mu[1])
    print(f'\nx0={x0}')
    if np.isnan(x0) or np.isinf(x0) or x0 == 0:
        print(f'\nx0={x0}')
    x0p = np.sqrt(1 + w112/cov_matrix[0,0] * alpha[0]**2)*x0
    if np.isnan(x0p) or np.isinf(x0p) or x0p == 0:
        print(f'\nx0p={x0p}')
    pdf = norm.pdf((x-xi_1)/np.sqrt(w112))
    cdf = norm.cdf(alpha[0]*np.sqrt(cov_matrix[0,0])*(x-xi_1) + x0p)
    print(f'\ncdf\n: {cdf}')
    print('\n', alpha[0]*np.sqrt(cov_matrix[0,0])*(x-xi_1) + x0p)
    print(f'\nalpha[0]: {alpha[0]}\ncov_matrix[0,0]: {cov_matrix[0,0]}\nxi_1: {xi_1}\nx0p: {x0p}\n')
    conditional_joint_skn = pdf * cdf / norm.cdf(x0)
    return conditional_joint_skn

def neg_log_likelihood_BN(params, data):
    mean = params[:2]
    L = np.zeros((2, 2))
    L[0, 0] = params[2]
    L[1, 0] = params[3]
    L[1, 1] = params[4]
    cov = np.dot(L, L.T)  # Construct the covariance matrix
    try:
        np.linalg.cholesky(cov)  # Check if the matrix is positive semidefinite
        return -np.sum(mvn.logpdf(data, mean=mean, cov=cov))
    except np.linalg.LinAlgError:
        return np.inf

def conditional_normal_sample(y, mu, cov):
    mu_1 = mu[0]
    mu_2 = mu[1]
    sgm_1 = np.sqrt(cov[0,0])
    sgm_2 = np.sqrt(cov[1,1])
    rho = cov[0,1] / (sgm_1*sgm_2)
    mean_cond_dist = mu_1 + sgm_1/sgm_2 * rho * (y - mu_2)
    var_cond_dist = (1 - rho**2) * cov[0,0]
    return mean_cond_dist, var_cond_dist

def callback_joint(x, iteration):
    mean = x[:2]
    a = x[5:]
    cov_matrix = np.array([[x[2],x[3]], [x[3],x[4]]])
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write(f"\nIteration: {iteration}")
        file.write(f"\nCurrent solution\n: Mean:\n{mean}\nAlpha:\n{a}\nCov Matrix:\n{cov_matrix}")

def create_callback_joint():
    iteration = 0
    def callback_closure_joint(x, *args):
        if len(args) == 1:
            state = args[0]
        nonlocal iteration
        iteration += 1
        callback_joint(x,iteration)
    return callback_closure_joint

def constraint_fun4(x):
    return -x[5]

def constraint_fun5(x):
    cov_matrix = np.array([[x[2],x[3]], [x[3],x[4]]])
    eigenvals = np.linalg.eigvals(cov_matrix)
    return np.min(eigenvals)

def joint_distribution_fit(i_initial_m, i_spoof, i_initial_p):
    nans = np.where(np.isnan(i_spoof))[0]
    i_spoof[nans] = 0
    nans = np.where(np.isnan(i_initial_m))[0]
    i_initial_m[nans] = 0
    i_initial_p[nans] = 0


    data_bn = np.column_stack((i_initial_m, i_initial_p))

    mask = i_initial_m != i_spoof
    i_spoof = i_spoof[mask]
    i_initial_p = i_initial_p[mask]
    data_skn = np.column_stack((i_spoof, i_initial_p))

    initial_params_BSKN = np.array([0, 0, 1, -0.7, 1, 0, 0])

    callback_func_skn = create_callback_joint()
    constraint_skn = [{'type': 'ineq', 'fun': constraint_fun5}]

    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write('\nBivariate Skew Normal Distribution fit')
    # mu = data_skn.mean(axis=0)
    # cov_matrix = np.cov(data_skn, rowvar=False)
    # params_skn = np.array([mu[0], mu[1], cov_matrix[0,0], cov_matrix[0,1], cov_matrix[1,1]])
    result_BSKN = minimize(multivariate_skewnorm, initial_params_BSKN, args=(data_skn,), method='SLSQP', constraints=constraint_skn, options={'maxiter': 1000})
    cov_matrix = np.array([[result_BSKN.x[2],result_BSKN.x[3]], [result_BSKN.x[3],result_BSKN.x[4]]])
    min_eig = np.min(np.linalg.eigvals(cov_matrix))
    if min_eig < 0:
        cov_matrix -= 10*min_eig * np.eye(*cov_matrix.shape)
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        # file.write(f'\n{params_skn}')
        file.write(f'\n{[result_BSKN.x[0], result_BSKN.x[1], cov_matrix[0,0], cov_matrix[0,1], cov_matrix[1,1], result_BSKN.x[5], result_BSKN.x[6]]}')
        file.write(f'\n{result_BSKN}')

    # You can use the maximum likelihood estimation
    # Anyway it turns out that in this case the mean and the covariance matrix estimator are the mean and the covariance
    # matrix of the data. See https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Maximum_likelihood_estimation
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write('\nBivariate Normal Distribution fit')
    
    mean = data_bn.mean(axis=0)
    covariance = np.cov(data_bn, rowvar=False)
    params_BN = np.array([mean[0], mean[1], covariance[0,0], covariance[0,1], covariance[1,1]])
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write(f'\n{params_BN}')
    
    return [params_BN, result_BSKN.x]

def Wasserstein_distance(p, u, v):
    '''This function computes the Wasserstein p-distance between two vectors u and v.

    Parameters
    ----------
    p : int
        Order of the distance.
    u : numpy array
        First vector.
    v : numpy array
        Second vector.
    
    Returns
    -------
    d : float
        Wasserstein p-distance between u and v.'''

    #For the WK-Means we exploit the Wasserstein distance
    u.sort()
    v.sort()
    n = len(u)
    d = 0 #Initialize variable
    for j in range(n): #Compute the distance
        d += abs(u[j] - v[j])**p
    d /= n #Normalize by the sample size
    return d

def spoofing_monitoring(params, num_previous_mo, i_initial_m, i_initial_p, num_quantiles):
    '''This function takes as input the parameters of the fitted distributions and the number of previous
    market orders to consider and returns the Wasserstein distance between the fitted distributions and the
    empirical ones (see the original paper https://arxiv.org/pdf/2009.14818v2.pdf).
    
    Parameters
    ----------
    params : list
        List containing the parameters of the fitted distributions.
    num_previous_mo : int
        Number of previous market orders to consider.
    i_initial_m : numpy array
        Array containing the initial imbalances for the market orders.
    i_initial_p : numpy array
        Array containing the imbalances after each market orders.
    num_quantiles : int
        Number of quantiles to consider.
        
    Returns
    -------
    mean_wasserstein_distance_normal : float
        Mean Wasserstein distance between the empirical distribution of the imbalance just before a market order
        and the fitted normal distribution.
    mean_wasserstein_distance_skewnormal : float
        Mean Wasserstein distance between the empirical distribution of the imbalance just before a market order
        and the fitted skew normal distribution.'''

    # Extract the fitted parameters     
    mu_skn = params[1][:2]
    cov_matrix_skn = np.array([[params[1][2],params[1][3]], [params[1][3],params[1][4]]])
    alpha_skn = params[1][5:7]

    estimated_mean = params[0][:2]
    cov_matrix_n = np.array([[params[0][2],params[0][3]], [params[0][3],params[0][4]]])

    # Consider the short term joint distribution of the imbalance just before and after a market order.
    # This meeans that I have to consider a number equal to num_previous_mo of market orders before the
    # market order indexed by idxs_marketorder.
    mean_wasserstein_distance_normal = []
    mean_wasserstein_distance_skewnormal = []
    # i_initial_m.shape[0]
    for t in range(num_previous_mo, i_initial_m.shape[0]):
    # for t in tqdm(range(100), desc='Considering every market order'):
        # Select the short span imbalances
        i_m_N = i_initial_m[t: t - num_previous_mo: -1][::-1]
        i_p_N = i_initial_p[t: t - num_previous_mo: -1][::-1]

        # Divide i_p_N into num_quantiles bins and store the 
        # indexes of the elements of i_p_N for each bin.
        equidistant_quantiles_ip = partition_into_quantiles(i_p_N, num_quantiles)
        idxs_quantiles_ip = []
        for quantile_idx in range(len(equidistant_quantiles_ip)):
            indxs = [np.where(i_p_N == equidistant_quantiles_ip[quantile_idx][i])[0][0] for i in range(equidistant_quantiles_ip[quantile_idx].shape[0])]
            # The following is a list of (num_quantiles) lists, where each list corresponds
            # to the quantile k, and it is the set of all the indexes I of i_m_p such that i_m_p[I] 
            # are the values of the k-th quantile.
            idxs_quantiles_ip.append(indxs)

        # Take the midpoint for each quantile
        mid_ip_quantile = np.array([np.sum(i_p_quantile)*(num_quantiles/num_previous_mo) for i_p_quantile in equidistant_quantiles_ip])
        np.save(f'output/mid_ip_quantile.npy', mid_ip_quantile)

        # I have to take a random sample of size num_previous_mo/num_quantiles for i-^(N,L) and i_spoof^(N,L)
        # from the fitted distributions where I fix the value of i+ equal to the mid point of each quantile.
        wasserstein_distance_normal = []
        wasserstein_distance_skewnormal = []

        if t % 1000 == 0:
            with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
                file.write(f'\n{(t+1)/i_initial_m.shape[0] * 100:.3f}%')

        for quantile_idx in range(len(equidistant_quantiles_ip)):
            n_sample = equidistant_quantiles_ip[quantile_idx].shape[0]
            cond_skn_pdf = conditional_joint_skn_pdf(mid_ip_quantile[quantile_idx], mu_skn, cov_matrix_skn, alpha_skn, n_sample)
            # print(f'cond_skn_pdf: {cond_skn_pdf}')
            # if np.isnan(cond_skn_pdf).any() or np.isnan(cond_skn_pdf.sum()) or cond_skn_pdf.sum() == 0:
            #     print(f'\nt: {t}')
            #     print(f'\nquantile: {quantile_idx}')
            #     print(f'\nmidpoint: {mid_ip_quantile[quantile_idx]}')
            #     print(f'cond_skn_pdf: {cond_skn_pdf}')
            mean_cond_dist, var_cond_dist = conditional_normal_sample(mid_ip_quantile[quantile_idx], estimated_mean, cov_matrix_n)
            # mean_cond_dist1, var_cond_dist1 = conditional_normal_sample(mid_ip_quantile[quantile_idx], mu_skn, cov_matrix_skn)
            # with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
            #     file.write(f'\nquantile: {quantile_idx}')
            #     file.write(f'\nmidpoint: {mid_ip_quantile[quantile_idx]}')
            #     file.write(f'\nmean_cond_dist: {mean_cond_dist}')
            #     file.write(f'\nvar_cond_dist: {var_cond_dist}')
            #     file.write(f'\nmean_cond_dist1: {mean_cond_dist1}')
            #     file.write(f'\nvar_cond_dist1: {var_cond_dist1}')
            w_n, w_skn = [], []
            for _ in range(25):
                normal_samples = np.random.normal(mean_cond_dist, np.sqrt(var_cond_dist), size=n_sample)
                # skewnorm_samples = np.random.normal(mean_cond_dist1, np.sqrt(var_cond_dist1), size=n_sample)
                skewnorm_samples = np.random.choice(np.linspace(0, 1, n_sample), p=cond_skn_pdf/cond_skn_pdf.sum(), size=n_sample)

                w_n.append(Wasserstein_distance(2, normal_samples, i_m_N[idxs_quantiles_ip[quantile_idx]]))
                w_skn.append(Wasserstein_distance(2, skewnorm_samples, i_m_N[idxs_quantiles_ip[quantile_idx]]))
            wasserstein_distance_normal.append(np.mean(w_n))
            wasserstein_distance_skewnormal.append(np.mean(w_skn))
        # Compute the mean of the Wasserstein distances over all the quantiles.
        mean_wasserstein_distance_normal.append(np.mean(wasserstein_distance_normal))
        mean_wasserstein_distance_skewnormal.append(np.mean(wasserstein_distance_skewnormal))

        if t % 5000 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
            ax.plot(mean_wasserstein_distance_normal, 'k', label='W($i_{-}$, $\hat{i}_{-}^N|\hat{i}_{+}^N$)')
            ax.set_xlabel('t')
            ax.set_ylabel('Wasserstein distance')
            ax.plot(mean_wasserstein_distance_skewnormal, 'r', label='W($i_{spoof}$, $\hat{i}_{-}^N|\hat{i}_{+}^N)$')
            ax.legend()
            plt.savefig(f'images/wasserstein_distance_{info[0]}_{info[1]}_{N}_{job_number}.png')

    return mean_wasserstein_distance_skewnormal, mean_wasserstein_distance_normal
    
def surface_joint_plot(i_spoof, i_initial_m, i_initial_p, params):

    params_skn = params[1]
    params_n = params[0]

    # Create scatter plot
    plt.figure()
    plt.scatter(i_initial_m, i_initial_p, c='k', s=1.2, alpha=0.3)
    # Fitted parameters for the joint normal distribution
    mean = params_n[:2]
    covariance = np.array([[params_n[2], params_n[3]], [params_n[3], params_n[4]]])
    # Generate points for surface curves
    x1_range = np.linspace(min(i_initial_m), max(i_initial_m), 1000)
    x2_range = np.linspace(min(i_initial_p), max(i_initial_p), 1000)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    positions = np.dstack([X1, X2])
    Z = mvn.pdf(positions, mean=mean, cov=covariance)
    # Plot surface curves as an image
    plt.imshow(Z, extent=[min(i_initial_m), max(i_initial_m), min(i_initial_p), max(i_initial_p)], origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(label='Density')
    Z = Z.reshape(X1.shape)
    # Plot surface curves
    plt.contour(X1, X2, Z, cmap='viridis', alpha=0.8)
    plt.xlabel(r'$i_-$')
    plt.ylabel(r'$i_+$')
    plt.title('Scatter Plot with Joint Normal Distribution')
    plt.savefig(f'images/surface_joint_normal.png')

    mean = params[:2]
    a = params[5:]
    cov_matrix = np.array([[params_skn[2],params_skn[3]], [params_skn[3],params_skn[4]]])

    x1_range = np.linspace(min(i_initial_m), max(i_initial_m), 1000)
    x2_range = np.linspace(min(i_initial_p), max(i_initial_p), 1000)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    positions = np.dstack([X1, X2])

    pdf_func, logpdf_func = multivariate_skewnorm(params_skn, positions)
    Z = pdf_func(positions)
    plt.scatter(i_spoof, i_initial_p, c='k', s=1.5, alpha=0.1)
    plt.imshow(Z, extent=[min(i_spoof), max(i_spoof), min(i_initial_p), max(i_initial_p)], origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(label='Density')
    norm_pdf = mvn.pdf(positions, mean, cov_matrix)

    plt.contour(X1, X2, Z, cmap='viridis', alpha=0.8)
    plt.contour(X1, X2, 2*norm_pdf, cmap='viridis', linestyles='dashed', alpha=1)
    plt.title('Scatter Plot with Joint Skew Normal Distribution')
    plt.xlabel(r'$i^{spoof}$')
    plt.ylabel(r'$i_+$')
    
    plt.savefig(f'images/surface_joint_skew_normal.png')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script performs the analysis of the LOBSTER data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-TSLA", "--TSLA", action='store_true')
    parser.add_argument("-MSFT", "--MSFT", action='store_true')
    parser.add_argument("-imb", "--imbalance_plot", action='store_true', help='Plot the imbalance distribution')
    parser.add_argument("-pp", "--pre_proc", action='store_true', help='Perform data preprocessing')
    parser.add_argument("-dq", "--dq_dist", action='store_true', help='Compute the distribution of dq')
    parser.add_argument("-dp", "--dp_dist", action='store_true', help='Compute the distribution of dp')
    parser.add_argument("-lob", "--lob_reconstruction", action='store_true', help='Reconstruct the limit order book')
    parser.add_argument("-f", "--freq", action='store_true', help='Compute the optimal frequency of sampling')
    parser.add_argument("-j", "--joint_imbalance", action='store_true', help='Compute the joint imbalance distribution')
    parser.add_argument("-i", "--i_spoof", action='store_true', help='Compute the i-spoofing')
    parser.add_argument("-m", "--monitoring", action='store_true', help='Monitoring the spoofing')
    parser.add_argument("-jp", "--joint_plots", action='store_true', help='Plots of the N and SKN surface levels of the empirical joint distribution')
    parser.add_argument("-mv", "--mean_volumes", action='store_true', help='Compute the mean volumes')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    np.random.seed(666)
    job_number = os.getenv("PBS_JOBID")

    tick = 100 # Set the tick size. For TSLA (MSFT) the tick size in 2013(2018) was 0.01 USD (100 in the data)
    info = [['TSLA', '2015-01-01', 'TSLA_2015-01-01_2015-01-31_10'], ['MSFT', '2018-04-01', 'MSFT_2018-04-01_2018-04-30_5']]
    if args.TSLA:
        info = info[0]
        depth = 6
    elif args.MSFT:
        info = info[1]
        depth = 4

    if args.pre_proc:
        logging.info('Performing data preprocessing')
        N_days = int(input('Number of days to consider: '))
        paths_lob = []
        paths_msg = []
        if args.TSLA:
            folder_path = 'data/TSLA_2015-01-01_2015-01-31_10'
            filenames = os.listdir(folder_path)
            for file in filenames:
                if 'orderbook' in file:
                    paths_lob.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
                elif 'message' in file:
                    paths_msg.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
        elif args.MSFT:
            folder_path = 'data/MSFT_2018-04-01_2018-04-30_5'
            filenames = os.listdir(folder_path)
            for file in filenames:
                if 'orderbook' in file:
                    paths_lob.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
                elif 'message' in file:
                    paths_msg.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
        elif args.AMZN:
            folder_path = 'data/AMZN_2012-06-21_2012-06-21_10'
            filenames = os.listdir(folder_path)
            for file in filenames:
                if 'orderbook' in file:
                    paths_lob.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
                elif 'message' in file:
                    paths_msg.append(f'/home/ddinosse/MMforQuantFin/{folder_path}/{file}')
        
        paths_lob.sort()
        paths_msg.sort()
        orderbook, message, event_per_day = data_preproc(paths_lob, paths_msg, N_days)
        np.save(f'output/events_per_day_{info[0]}_{info[1]}', event_per_day)
        orderbook.to_parquet(f'{info[0]}_orderbook_{info[1]}.parquet')
        message.to_parquet(f'{info[0]}_message_{info[1]}.parquet')
        logging.info('Data preprocessing completed. Exiting...')
        exit()

    # Select the number of events to consider
    N = int(input(f'Number of days to consider: '))

    # Read the data unsliced
    event_per_day = np.load(f'output/events_per_day_{info[0]}_{info[1]}.npy')[:N]
    # Create a list of dataframes, one for each day
    orderbook = [pd.read_parquet(f'{info[0]}_orderbook_{info[1]}.parquet')[event_per_day[:i].sum():event_per_day[:i+1].sum()].reset_index() for i in range(event_per_day.shape[0])]
    message = [pd.read_parquet(f'{info[0]}_message_{info[1]}.parquet')[event_per_day[:i].sum():event_per_day[:i+1].sum()].reset_index() for i in range(event_per_day.shape[0])]
    # period = [message['Time'].iloc[0].strftime('%Y-%m-%d'),  message['Time'].iloc[-1].strftime('%Y-%m-%d')]

    var = int(input('Recompute (1) the initial quantities or load (2) them?: '))
    if var == 1:
        # This vectors start from the very first element of the orderbook
        bid_prices, bid_volumes, ask_prices, ask_volumes = [], [], [], []
        for lob in orderbook:
            bid_price, bid_volume, ask_price, ask_volume = prices_and_volumes(lob)
            bid_prices.append(bid_price)
            bid_volumes.append(bid_volume)
            ask_prices.append(ask_price)
            ask_volumes.append(ask_volume)
        np.save(f'output/ask_prices_{info[0]}_{info[1]}_{depth}', ask_prices)
        np.save(f'output/ask_volumes_{info[0]}_{info[1]}_{depth}', ask_volumes)
        
        volume_ask_imb, volume_bid_imb =  [], []
        for i in range(len(ask_prices)):
            vol_ask_imb, vol_bid_imb = volumes_for_imbalance(ask_prices[i], bid_prices[i], ask_volumes[i], bid_volumes[i], depth)
            volume_ask_imb.append(vol_ask_imb)
            volume_bid_imb.append(vol_bid_imb)
        np.save(f'output/volume_ask_imb_{info[0]}_{info[1]}_{N}_{depth}', volume_ask_imb)
        np.save(f'output/volume_bid_imb_{info[0]}_{info[1]}_{N}_{depth}', volume_bid_imb)

        logging.info('\nComputing the optimal sampling frequency...')
        '''Compute the difference between the ask prices sampled every f events.
        Remember that you want to consider the price change between t and t+f
        and the value of the imbalance at time t. The first element of ask_prices_fdiff
        is the price change between the initial time and the f events after. Since I do not
        have the value of the imbalance at the initial time, I have to skip it. Indeed,
        the first element of the imbalance is the one computed considering the f events
        after the initial time.'''
        frequencies = []
        ask_prices_fdiffs = []
        for i in range(len(ask_prices)):
            f = time_interval_ret_dist(ask_prices[i][:,0], tick)
            best_ask_price = ask_prices[i][:,0] / tick
            diff_sampled = np.diff(best_ask_price[::f]).astype(int)
            ask_prices_fdiffs.append(diff_sampled[1:])
            frequencies.append(f)
        np.save(f'output/frequencies_{info[0]}_{info[1]}_{N}_{depth}', frequencies)
        np.save(f'output/ask_prices_fdiffs_{info[0]}_{info[1]}_{N}_{depth}', ask_prices_fdiffs)

        # These vectors start from the f-th element of the orderbook.
        # Moreover, they are evaluated with the volumes needed for the imbalance.
        bid_volumes_fsummed, ask_volumes_fsummed = [], []
        for i in range(len(bid_volumes)):
            bid_vol_fsummed, ask_vol_fsummed = volumes_fsummed(frequencies[i], depth, volume_ask_imb[i], volume_bid_imb[i])
            bid_volumes_fsummed.append(bid_vol_fsummed)
            ask_volumes_fsummed.append(ask_vol_fsummed)
        np.save(f'output/bid_volumes_fsummed_{info[0]}_{info[1]}_{N}_{depth}', bid_volumes_fsummed)
        np.save(f'output/ask_volumes_fsummed_{info[0]}_{info[1]}_{N}_{depth}', ask_volumes_fsummed)
    else:
        ask_prices = np.load(f'output/ask_prices_{info[0]}_{info[1]}.npy', allow_pickle=True)
        ask_volumes = np.load(f'output/ask_volumes_{info[0]}_{info[1]}.npy', allow_pickle=True)
        volume_ask_imb = np.load(f'output/volume_ask_imb_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        volume_bid_imb = np.load(f'output/volume_bid_imb_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        ask_prices_fdiffs = np.load(f'output/ask_prices_fdiffs_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        bid_volumes_fsummed = np.load(f'output/bid_volumes_fsummed_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        ask_volumes_fsummed = np.load(f'output/ask_volumes_fsummed_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        frequencies = np.load(f'output/frequencies_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)

    logging.info('\nData loaded: {}.\nFrequencies: {}.\nDepth: {}.\nPeriod: {} days\n'.format(info, frequencies, depth, N))
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write('\nAction: {}.\nPid: {}.\nStart: {}.\nData loaded: {}.\nNumber of days: {}. \
                   \nFrequencies: {}.\nDepth: {}.'.format(sys.argv[-1], job_number, datetime.datetime.now(), info, N, frequencies, depth))

    if args.lob_reconstruction:
        lob_reconstruction(N, tick, bid_prices, bid_volumes, ask_prices, ask_volumes)
        lob_video('lob_snapshots', info)

    if args.dq_dist:
        var = int(input('Recompute (1) or load (2) the dq?: '))

        if var == 1:
            logging.info('\nComputing the dq distribution...')
            dqs = []
            MO_vols = []
            # Note that each dataframe starts from 0
            idxs_m = []
            idxs_p = []
            for i in range(len(message)):
                executions = executions_finder(message[i], frequencies[i])
                dqs.append(dq_dist(executions, orderbook[i], depth))

                MO_vol, idx_m, idx_p = MO_volumes(message[i], frequencies[i])
                MO_vols.append(MO_vol)
                idxs_m.append(idx_m)
                idxs_p.append(idx_p)
            dq = np.concatenate(dqs)
            value, count = np.unique(dq, return_counts=True)
            x = np.arange(-depth, depth+1)
            mask = np.in1d(x, value)
            y = np.zeros_like(x)
            y[mask] = count
            y = y / count.sum()
            np.save(f'output/dq_{info[0]}_{N}_{job_number}.npy', y)
            
        elif var == 2:
            dq = np.load(f'dq_{info[0]}_{info[1]}_{N}_{depth}.npy')

        # value, count = np.unique(dq, return_counts=True)
        # x = np.arange(-depth, depth+1)
        # mask = np.in1d(x, value)
        # y = np.zeros_like(x)
        # y[mask] = count
        # y = y / count.sum()
        # np.save(f'output/dq_{info[0]}_{N}_{job_number}.npy', y)
    
        # MO_volumes_mean = MO_vols.mean()
        # with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        #     file.write('\n<V_mo>\n'.format(MO_volumes_mean))

        plt.figure(tight_layout=True, figsize=(8,5))
        plt.bar(np.arange(-depth, depth+1), y, color='green', edgecolor='black')
        plt.title(fr'$dq$ for {info[0]}')
        plt.xlabel('Ticks deviation')
        plt.ylabel('Frequency')
        plt.savefig(f'images/dq_{info[0]}_{N}_{job_number}.png')
    
    if args.dp_dist:
        bifurcation = int(input('Recompute (1) or chi-square (2) the dp+ distibrution?:\n'))
        if bifurcation == 1:
            M = int(depth)*2 + 1
            initial_params = np.random.uniform(0, 1, M+depth)
            obj_fun = []

            optimizer = 'SLSQP'
            constraint1 = {'type': 'eq', 'fun': constraint_fun1, 'args': (depth,)}
            constraint2 = {'type': 'eq', 'fun': constraint_fun2, 'args': (depth,)}
            constraints = [constraint1, constraint2] # constraint3]
            bounds = [(0.0001,1) for i in range(M+depth)]

            callback_func = create_callback(M, ask_prices_fdiffs, bid_volumes_fsummed, ask_volumes_fsummed, frequencies, depth)

            res = minimize(dp_distribution_objective, initial_params, args=(M, ask_prices_fdiffs, bid_volumes_fsummed, ask_volumes_fsummed, frequencies, depth),  \
                        constraints=constraints, method=optimizer, bounds=bounds, \
                        callback=callback_func)

            with open(f'opt_{job_number}.txt', 'a', encoding='utf-8') as file:
                file.write(f"\nFINAL RESULT\n: {res}")
            print(res.x)
            print("Initial parameters:\n", initial_params)
            np.save(f'output/dp_p_{info[0]}_{info[1]}_{N}_{job_number}.npy', res.x[:M])
            np.save(f'output/ws_{info[0]}_{info[1]}_{N}_{job_number}.npy', res.x[M:])
            imbalance = []
            for day in range(len(ask_prices_fdiffs)):
                f = frequencies[day]
                # Compute the imbalance for each days in N days
                l = avg_imbalance_faster(bid_volumes_fsummed[day], ask_volumes_fsummed[day], res.x[M:], f)
                imbalance.append(l)
            imb = np.concatenate(imbalance)
            np.save(f'output/imb_{info[0]}_{info[1]}_{N}.npy', imb)

            plt.figure(tight_layout=True, figsize=(5,5))
            plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), res.x[:M], color='green', edgecolor='black')
            plt.title(r'$dp^+$')
            plt.savefig(f'images/dp_p_{info[0]}_{info[1]}_{N}_{optimizer}_{job_number}.png')

            plt.figure(tight_layout=True, figsize=(13,5))
            plt.hist(imb, bins=100, histtype='bar', density=True, edgecolor='black', alpha=0.5, label=f'{res.x[M:]}')
            plt.xlabel('Imbalance')
            plt.legend()
            plt.title(f'Imbalance distribution for {info[0]}')
            plt.savefig(f'images/imbalance_{info[0]}_{info[1]}_{res.x[M:]}_{job_number}.png')
            plt.show()

        if bifurcation == 2:
            logging.info('Computing the chi-square test...')
            weights = np.load(f'output/ws_{info[0]}_{info[1]}_{N}.npy')
            imbalance = []
            for day in range(len(ask_prices_fdiffs)):
                f = frequencies[day]
                # Compute the imbalance for each days in N days
                l = avg_imbalance_faster(bid_volumes_fsummed[day], ask_volumes_fsummed[day], weights, f)[::f]
                imbalance.append(l)
            imb = np.concatenate(imbalance)
            quantiles = partition_into_quantiles(imb, 20) # Bucket the imbalance into 20 quantiles
            # Take the indexes of the imbalance that are in each quantile and consider the price change at the f-th event after that value of the imbalance
            # (so I have the pair (x_m, i_m) where x_m represents the price change "influenced" by i_m (so I see i_m at time t and check the price change at time t+f)
            # and compute the probability of that price change given the imbalance.
            imb_indxs = np.array([np.array([np.where(imb == i)[0][0] for i in quantiles[j]], dtype=object) for j in tqdm(range(quantiles.shape[0]), desc='idxs imb for each quantile')], dtype=object) # Take the indexes of the imbalance that are in each quantile
            imb_indxs = np.array([np.sort(imb_indxs[i]) for i in range(quantiles.shape[0])], dtype=object) # For each quantile I sort the indexes
            ask_prices_fdiff = np.concatenate(ask_prices_fdiffs)
            p_changes_quantile = np.array([np.array([ask_prices_fdiff[i] for i in imb_indxs[j][:-1]]) for j in tqdm(range(quantiles.shape[0]), desc='p change for each quantile')]) # Take the price change at the f-th event after that value of the imbalance
            
            # Compute the empirical probability distribution of the price change given the imbalance for each quantile
            dp_imb_quantile = np.zeros(shape=(quantiles.shape[0], depth*2+1))
            for i in tqdm(range(quantiles.shape[0]), desc='dp emp for each quantile'):
                changes = p_changes_quantile[i][p_changes_quantile[i] <= depth]
                changes = changes[changes >= -depth]
                value, count = np.unique(changes, return_counts=True)
                x = np.arange(-depth, depth+1)
                mask = np.isin(x, value)
                y = np.zeros_like(x)
                y[mask] = count
                dp_imb_quantile[i, :] = y / count.sum()
            np.save(f'output/dp_quantemp_{info[0]}_{info[1]}_{N}.npy', dp_imb_quantile)
            
            # Compute the theoretical probability distribution of the price change given the imbalance for each quantile
            # The imbalance is taken as the mid value of each quantile
            mid_quantiles = np.array([np.mean(quantiles[i]) for i in range(quantiles.shape[0])])  # Take the mid value of each quantile
            dp_p = np.load(f'output/dp_p_{info[0]}_{info[1]}_{N}.npy') # Load the distribution of dp+
            dp_imb_theory = np.zeros(shape=(quantiles.shape[0], depth*2+1))
            for i in tqdm(range(quantiles.shape[0]), desc='theoretical dp for each quantile'):
                dp_imb_theory[i, :] = mid_quantiles[i] * dp_p + (1 - mid_quantiles[i]) * dp_p[::-1]
            np.save(f'output/dp_quanttheory_{info[0]}_{info[1]}_{N}.npy', dp_imb_theory)
            create_bar_plot(dp_imb_quantile[0], dp_imb_theory[0], np.arange(-depth, depth+1), ['Empirical', r'$\bar{i}dp^{+} + (1-\bar{i})dp^{-}$'], title=r'$i$ small' )
            create_bar_plot(dp_imb_quantile[-1], dp_imb_theory[-1], np.arange(-depth, depth+1), ['Empirical', r'$\bar{i}dp^{+} + (1-\bar{i})dp^{-}$'], title=r'$i$ large')
            
            # Perform a chi-square test to check if the empirical distribution is compatible with the theoretical one
            p_values = np.zeros(quantiles.shape[0])
            for i in tqdm(range(quantiles.shape[0]), desc='chi-square for each quantile'):
                p_values[i] = stats.chisquare(dp_imb_quantile[i], dp_imb_theory[i], ddof=(2*depth+1)-depth+1)[1]
                print(f'p-value for quantile {i}: {p_values[i]}')
            np.save(f'output/p_values_{info[0]}_{info[1]}_{N}.npy', p_values)

            plt.figure(tight_layout=True, figsize=(8,5))
            plt.bar(np.arange(quantiles.shape[0]), p_values, color='green', edgecolor='black')
            plt.title('Chi-square test')
            plt.xlabel('Quantile')
            plt.ylabel('p-value')
            plt.savefig(f'images/chi_square_{info[0]}_{info[1]}_{N}.png')

    if args.imbalance_plot:
        weights = np.load(f'/home/ddinosse/MMforQuantFin/output/ws_{info[0]}_{info[1]}_{N}.npy')
        boot = int(input('Bootstrap (1) or not (2)?: '))

        imbalance, parameters, errors = [], [], []
        for i in range(len(volume_bid_imb)):
            f = frequencies[i]
            imb = avg_imbalance_faster(bid_volumes_fsummed[i], ask_volumes_fsummed[i], weights, frequencies[i])[::f]
            imbalance.append(imb)
        imbalance = np.concatenate(imbalance)
        
        if boot == 1:
            var = int(input('Recompute (1) or load (2) the errors?: '))
            if var == 1:
                n_iterations = 100  # Number of bootstrap iterations
                n_samples = len(imb)  # Number of samples in your data

                # Initialize arrays to store parameter values from each iteration
                shape_samples = np.zeros(n_iterations)
                loc_samples = np.zeros(n_iterations)
                scale_samples = np.zeros(n_iterations)

                # Perform bootstrapping and refit the skew normal distribution in each iteration
                for i in tqdm(range(n_iterations), desc='Computing standard errors'):
                    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    bootstrap_sample = imb[bootstrap_indices]
                    shape_samples[i], loc_samples[i], scale_samples[i] = stats.skewnorm.fit(bootstrap_sample)

                # Calculate the standard errors of the parameters
                shape_error = np.std(shape_samples)
                loc_error = np.std(loc_samples)
                scale_error = np.std(scale_samples)
                np.save(f'output/imb_{info[0]}_{info[1]}_{N}_errors', np.array([shape_error, loc_error, scale_error]))
            else:
                shape_error, loc_error, scale_error = np.load(f'output/imb_{info[0]}_{info[1]}_{N}_errors.npy')
        else:
            shape_error, loc_error, scale_error = 0, 0, 0

        x = np.linspace(np.min(imbalance), np.max(imbalance), 500)
        shape, loc, scale = stats.skewnorm.fit(imbalance)
        pdf_sn = stats.skewnorm.pdf(x, shape, loc, scale)
        np.save(f'output/imb_{info[0]}_{info[1]}_{N}_params.npy', np.array([shape, loc, scale]))
        plt.figure(tight_layout=True, figsize=(13,5))
        plt.plot(x, pdf_sn/pdf_sn.sum(), 'r-', label=fr'Skew Normal $\to$ $\alpha$: {shape:.3f}$\pm${(2 * shape_error):.3f}, $\mu$: {loc:.3f}$\pm${(2 * loc_error):.3f}, $\sigma$: {scale:.3f}$\pm${(2*scale_error):.3f}')
        plt.hist(imbalance, bins=100, histtype='bar', density=True, edgecolor='black', alpha=0.5)
        plt.xlabel('Imbalance')
        plt.legend()
        plt.title(f'Imbalance distribution for {info[0]}. Skew: {stats.skewnorm.stats(shape, loc, scale, moments="mvsk")[2]:.3f}')
        plt.savefig(f'images/imb_{info[0]}_{info[1]}_{weights}.png')
    
    if args.joint_imbalance:
        var = int(input('Recompute (1) or load (2) the imbalance?: '))

        if var == 1:
            weight = np.load(f'output/ws_{info[0]}_{info[1]}_{N}.npy')
            imb = avg_imbalance_faster(N, bid_volumes_fsummed, ask_volumes_fsummed, weight, f)
            np.save(f'output/imb_{info[0]}_{info[1]}_{N}_{job_number}.npy', imb)
            i_m_sell, i_p_sell, i_m_buy, i_p_buy = MO_imbalances(message, imb, N, f)

        elif var == 2:
            i_m_sell, i_p_sell, i_m_buy, i_p_buy = np.load(f'output/i_m_sell_{info[0]}_{info[1]}_{f}_{N}.npy'), \
                np.load(f'output/i_p_sell_{info[0]}_{info[1]}_{N}.npy'), \
                np.load(f'output/i_m_buy_{info[0]}_{info[1]}_{N}.npy'), \
                np.load(f'output/i_p_buy_{info[0]}_{info[1]}_{N}.npy')
        
        # Perform the Ljung Box test for absence of autocorrelation
        ljung_box_test_sell_m = sm.stats.acorr_ljungbox(i_m_sell)
        ljung_box_test_sell_p = sm.stats.acorr_ljungbox(i_p_sell)
        ljung_box_test_buy_m = sm.stats.acorr_ljungbox(i_m_buy)
        ljung_box_test_buy_p = sm.stats.acorr_ljungbox(i_p_buy)
        ljung_box_test = np.array([ljung_box_test_sell_m, ljung_box_test_sell_p, ljung_box_test_buy_m, ljung_box_test_buy_p])
        np.save(f'output/ljung_box_test_{info[0]}_{info[1]}_{f}_{N}_{job_number}.npy', ljung_box_test)

        plt.figure(tight_layout=True, figsize=(7,7))
        plt.scatter(i_m_sell, i_p_sell, s=13, c='green', edgecolors='black', alpha=0.65)
        plt.title(r'Joint imbalance distribution of $(i_{-}, i_{+})$ for sell MOs', fontsize=13)
        plt.xlabel(r'$i_{-}$', fontsize=13)
        plt.ylabel(r'$i_{+}$', fontsize=13)
        plt.savefig(f'images/joint_imb_sell_{info[0]}_{job_number}.png')

        plt.figure(tight_layout=True, figsize=(7,7))
        plt.scatter(i_m_buy, i_p_buy, s=13, c='green', edgecolors='black', alpha=0.65)
        plt.title(r'Joint imbalance distribution of $(i_{-}, i_{+})$ for buy MOs', fontsize=13)
        plt.xlabel(r'$i_{-}$', fontsize=13)
        plt.ylabel(r'$i_{+}$', fontsize=13)
        plt.savefig(f'images/joint_imb_buy_{info[0]}_{job_number}.png')

        '''The sequence of i_{-} and i_{+} are correlated, as shown in the autocorrelation function plot.
        This fact negates the idea to use the imbalance just before a market order to detect spoofing, since
        the difference between the imbalance just before and after a market order and the long run
        imbalance can be attributed to market conditions (the ones that generates the correlation).'''
        # What about the correlation between the imbalance and the price change? And with the direction of the trade?
        fig, ax = plt.subplots(2, 1, figsize=(13,7))
        fig.suptitle('Autocorrelation function of $i_{-}$ and $i_{+}$', fontsize=13)
        sm.graphics.tsa.plot_acf(i_m_sell, ax=ax[0], lags=100)
        sm.graphics.tsa.plot_acf(i_p_sell, ax=ax[1], lags=100)
        ax[0].set_title(r'$i_{-}(sell MO)$', fontsize=13)
        ax[1].set_title(r'$i_{+}(sell MO)$', fontsize=13)
        plt.savefig(f'images/autocorrelation_imip_{info[0]}_{job_number}.png')

        # Autocorrelation of i-|i+
        quantiles = partition_into_quantiles(i_p_sell, 100)
        for i in range(4):
            common_elements = np.intersect1d(i_p_sell, quantiles[i])
            indices = np.where(np.isin(i_p_sell, common_elements))
            
            sm.graphics.tsa.plot_acf(i_m_sell[indices], lags=100, title=f'Autocorrelation {i+1}th 100-quantile')
            plt.xlabel('Lags')
            plt.savefig(f'/home/danielemdn/Documents/MMforQuantFin/images/acf_tsla_{i+1}100quantile.png')
    
    if args.i_spoof:
        var = int(input('Recompute (1) or load (2) the imbalance, dq, H and i-,i+ indexes?:\n '))
        weights = np.load(f'output/ws_{info[0]}_{info[1]}_{N}.npy') # Load the weights evaluated via the optimization
        if var == 1:
            logging.info('Computing dq...')
            dqs = []
            for i in range(len(message)):
                executions = executions_finder(message[i], frequencies[i])
                dqs.append(dq_dist(executions, orderbook[i], depth))
            dq = np.concatenate(dqs)
            value, count = np.unique(dq, return_counts=True)
            x = np.arange(-depth, depth+1)
            mask = np.in1d(x, value)
            y = np.zeros_like(x)
            y[mask] = count
            y = y / count.sum()
            np.save(f'output/dq_{info[0]}_{N}.npy', y)

            logging.info('Computing H...')
            H_list = []
            # Note that each dataframe starts from 0
            idxs_m_list = []
            idxs_p_list = []
            for i in range(len(message)):
                H, idx_m, idx_p = MO_volumes(message[i], frequencies[i])
                H_list.append(H)
                idxs_m_list.append(idx_m)
                idxs_p_list.append(idx_p)
    
            np.save(f'output/H_{info[0]}_{info[1]}_{N}.npy', H_list)
            np.save(f'output/idxs_m_{info[0]}_{info[1]}_{N}.npy', idxs_m_list)
            np.save(f'output/idxs_p_{info[0]}_{info[1]}_{N}.npy', idxs_p_list)
            logging.info('\nThe distribution dp+ and the weights w are very costly to compute as they take several days\
                         to be estimated. Hence, it is better to load the values of dp+ and w. If \
                         you do not have them, run the script with the flag -dp (see -h for more info).')
        if var == 2:
            dq = np.load(f'output/dq_{info[0]}_{N}.npy')
            H_list = np.load(f'output/H_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
            idxs_m_list = np.load(f'output/idxs_m_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
            idxs_p_list = np.load(f'output/idxs_p_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)

        # Load the distribution of dp_p evaluated via the optimization
        dp_p = np.load(f'output/dp_p_{info[0]}_{info[1]}_{N}.npy')
        # dp_p = dp_p[::-1] # beh...
        logging.info('Computing i_spoof_k...')
        # Choose a startng point for the least square problem set to
        # solve the implicit cubic equation
        x0 = np.random.uniform(0,1)
        i_spoof_k, v_spoof, i_spoof, i_initial_m, i_initial_p, spoof_params = [], [], [], [], [], []
        for day in range(N):
            H = H_list[day]
            idxs_m = idxs_m_list[day]
            idxs_p = idxs_p_list[day]
            f = frequencies[day]
            vol_ask_imb = volume_ask_imb[day]
            vol_bid_imb = volume_bid_imb[day]
            i_spoof_k_day, v_spoof_day, i_spoof_day, i_initial_m_day, i_initial_p_day, spoof_params_day = i_spoofing(dq, dp_p, H, weights, depth, f,\
                            vol_ask_imb , vol_bid_imb, idxs_m, idxs_p, x0)
            i_spoof_k.append(i_spoof_k_day)
            v_spoof.append(v_spoof_day)
            i_spoof.append(i_spoof_day)
            i_initial_m.append(i_initial_m_day)
            i_initial_p.append(i_initial_p_day)
            spoof_params.append(spoof_params_day)

        for day in range(N):
            fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            fig.suptitle(r'Spoofing condition', fontsize=13)
            term = []
            Q, v, a_m, b_m, rho, mu_p = spoof_params[day]
            np.save(f'output/Q_{info[0]}_{info[1]}_{N}_{day+1}.npy', Q)
            np.save(f'output/v_{info[0]}_{info[1]}_{N}_{day+1}.npy', v)
            np.save(f'output/a_m_{info[0]}_{info[1]}_{N}_{day+1}.npy', a_m)
            np.save(f'output/b_m_{info[0]}_{info[1]}_{N}_{day+1}.npy', b_m)
            np.save(f'output/rho_{info[0]}_{info[1]}_{N}_{day+1}.npy', rho)
            np.save(f'output/mu_p_{info[0]}_{info[1]}_{N}_{day+1}.npy', mu_p)
            for k in range(depth):
                for t in range(rho.shape[0]):
                    y = 2 * rho[t] * weights[k] * mu_p * (1 - i_initial_m[day][t]) / i_initial_m[day][t] * i_spoof_k[day][t, k]**2 - (Q[k] * rho[t] + v[k])
                    term.append(y)
                term = np.array(term)
                mask = term > 0
                y = np.zeros_like(term)
                y[np.where(mask)[0]] = term[mask] # np.where(mask) returns the index where mask is True
                y[y==0] = np.nan
                ax[k].plot(term)
                ax[k].scatter(np.arange(rho.shape[0]), y, c='r', alpha=0.6)
                ax[k].set_title(f'Level {k}', fontsize=10)
                term = []
            plt.savefig(f'images/condition_spoof_{info[0]}_{info[1]}_{N}_{day+1}.png')
            plt.close()
        i_spoof = np.concatenate(i_spoof)
        i_initial_m = np.concatenate(i_initial_m)
        i_initial_p = np.concatenate(i_initial_p)
        # spoof_params = np.concatenate(spoof_params)
        v_spoof = np.concatenate(v_spoof)
        # Q, v, a_m, b_m, rho, mu_p = spoof_params
        np.save(f'output/i_spoof_k_{info[0]}_{info[1]}_{N}.npy', i_spoof_k)
        np.save(f'output/v_spoof_k_{info[0]}_{info[1]}_{N}.npy', v_spoof)
        # np.save(f'output/spoof_parameters_{info[0]}_{info[1]}_{N}.npy', np.array(spoof_params, dtype=object))
        np.save(f'output/i_spoof_{info[0]}_{info[1]}_{N}.npy', i_spoof)
        np.save(f'output/i_initial_m_{info[0]}_{info[1]}_{N}.npy', i_initial_m)
        np.save(f'output/i_initial_p_{info[0]}_{info[1]}_{N}.npy', i_initial_p)

        # Plot v_spoof
        fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
        fig.suptitle(r'Optimal strategy $v_k^{spoof}$')
        for i in range(depth):
            ax[i].plot(v_spoof[:,i], 'r', alpha=0.8)
            ax[i].set_title(rf'$Level_{i}$', fontsize=13)
        ax[-1].set_xlabel('t')
        plt.savefig(f'images/v_spoof_{info[0]}_{info[1]}_{N}_{job_number}.png')

        # Distribution of v_spoof for occupied levels
        for i in range(depth):
            if v_spoof[:,i].sum() != 0:
                plt.figure(figsize=(8,5), tight_layout=True)
                plt.hist(v_spoof[:,i], bins=50, color='green', edgecolor='black', alpha=0.7)
                plt.yscale('log')
                plt.xlabel('Volumes')
                plt.title(rf'Spoofing volume distribution for level {i}')
                plt.savefig(f'images/v_spoof_dist_{info[0]}_{info[1]}_{N}_{i}_{job_number}.png')

        # Plot i_spoof(i_initial_p)
        plt.figure(figsize=(10,7), tight_layout=True)
        plt.scatter(i_spoof, i_initial_p, c='green', s=1)
        plt.xlabel(r'$i_{spoof}$')
        plt.ylabel(r'$i_+$')
        plt.savefig(f'images/i_spoof_vs_ip_{info[0]}_{info[1]}_{N}_{job_number}.png')

         # Plot i_initial_m(i_initial_p)
        plt.figure(figsize=(10,7), tight_layout=True)
        plt.scatter(i_initial_m, i_initial_p, c='green', s=1)
        plt.xlabel(r'$i_-$')
        plt.ylabel(r'$i_+$')
        plt.savefig(f'images/im_vs_ip_{info[0]}_{info[1]}_{N}_{job_number}.png')

    if args.monitoring:
        logging.info('Monitoring the spoofing activity...')
        i_spoof = np.load(f'output/i_spoof_{info[0]}_{info[1]}_{N}.npy')
        # i_spoof = np.concatenate(np.load(f'output/i_spoof_k_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True))[:, 4]
        i_initial_m = np.load(f'output/i_initial_m_{info[0]}_{info[1]}_{N}.npy')
        i_initial_p = np.load(f'output/i_initial_p_{info[0]}_{info[1]}_{N}.npy')

        logging.info('\nFitting the bivariate normal and bivariate skew normal distributions')
        params = joint_distribution_fit(i_initial_m, i_spoof, i_initial_p)
        np.save(f'output/params_joint_fit_N_{info[0]}_{info[1]}_{N}.npy', params[0])
        np.save(f'output/params_joint_fit_SKN_{info[0]}_{info[1]}_{N}.npy', params[1])
        # Choose the fraction of market orders to collect before each MO and the number of bins.
        if args.TSLA:
            frac, n = 0.02, 20
        if args.MSFT:
            frac, n = 0.01, 25

        num_previous_mo = int(i_initial_m.shape[0] * frac)
        num_quantiles = n

        with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
            file.write('\nPrevious MO: {}.\nQuantiles: {}'.format(num_previous_mo, num_quantiles))

        wasserstein_distance_skewnormal, wasserstein_distance_normal = \
            spoofing_monitoring(params, num_previous_mo, i_initial_m, i_initial_p, num_quantiles)

        np.save(f'output/wasserstein_distance_{info[0]}_{info[1]}_{N}_{job_number}.npy', \
                [wasserstein_distance_normal, wasserstein_distance_skewnormal])

        # Plot wasserstein_distance_normal, wasserstein_distance_skewnormal on the same figure
        fig, ax = plt.subplots(1, 1, figsize=(10,7), tight_layout=True)
        ax.plot(wasserstein_distance_normal, 'k', label='W($i_{-}$, $\hat{i}_{-}^N|i_{+}^N$)')
        ax.set_xlabel(r'$t_{MO}$')
        ax.set_ylabel('Wasserstein distance')
        ax.plot(wasserstein_distance_skewnormal, 'r', label='W($i_{spoof}$, $\hat{i}_{-}^N|i_{+}^N)$')
        ax.legend()
        plt.savefig(f'images/wasserstein_distance_{info[0]}_{info[1]}_{N}_{job_number}.png')

    if args.joint_plots:
        i_spoof = np.load(f'output/i_spoof_{info[0]}_{info[1]}_{N}.npy')
        # i_spoof = np.concatenate(np.load(f'output/i_spoof_k_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True))[:, 2]
        i_initial_m = np.load(f'output/i_initial_m_{info[0]}_{info[1]}_{N}.npy')
        i_initial_p = np.load(f'output/i_initial_p_{info[0]}_{info[1]}_{N}.npy')
        params = np.load(f'output/params_joint_fit_{info[0]}_{info[1]}_{N}.npy', allow_pickle=True)
        surface_joint_plot(i_spoof, i_initial_m, i_initial_p, params)
    
    if args.mean_volumes:
        vols = np.zeros(shape=(depth,2))
        lob = pd.concat([orderbook[i] for i in range(len(orderbook))], ignore_index=True)
        for i in range(depth):
            vols[i] = np.mean(lob[f'Ask size {i+1}'].values), 2*np.std(lob[f'Ask size {i+1}'].values)
        
        plt.figure(figsize=(10,7), tight_layout=True)
        plt.bar(np.arange(1,depth+1), vols[:,0], yerr=vols[:,1], color='green', edgecolor='black', alpha=0.7)
        plt.xlabel('(Occupied) Levels')
        plt.ylabel('Mean volumes')
        plt.savefig(f'images/mean_volumes_{info[0]}_{info[1]}_{N}.png')
        with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
            file.write(f'\nMean and Std of volumes\n:{vols}')


    plt.show()
    with open(f'summary_{job_number}.txt', 'a', encoding='utf-8') as file:
        file.write('\nEnd: {}.'.format(datetime.datetime.now()))
    
    print('Last episode of One Piece was amazing!')