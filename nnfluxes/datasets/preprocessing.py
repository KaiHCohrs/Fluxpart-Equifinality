"""Functions for preparing the fluxnet dataset."""
import datetime
import math
import pathlib

import pandas as pd
import numpy as np

import nnfluxes.datasets.relevant_variables as relevant_variables



def preprocess(data):
    """Run a preprocessing pipeline which standardizes names, 
    keeps generally relevant variables, and computes some more.

    Args:`
        data (pd.DataFrame): Dataframe with all the FLUXNET data.
        
    Returns:
        data (pd.DataFrame): Dataframe with all the FLUXNET data after preprocessing.
    """
    
    #Run a data preprocessing pipeline
    data = unwrap_time(data)
    data = data.set_index('DateTime')
    data['DateTime'] = data.index
    data = standardize_column_names(data)
    data = data[list(set(data.columns)
                            & set(relevant_variables.variables))]
    data = NEE_quality_masks(data)
    
    data['SW_POT_sm'] = sw_pot_sm(data)
    data['SW_POT_sm_diff'] = sw_pot_sm_diff(data)
    data['CWD'] = wdefcum(data)
    data['SW_ratio'] = diffuse_to_direct_rad(data)
    data["doy"] = pd.to_datetime(data['DateTime']).dt.dayofyear
    data["tod"] = pd.to_datetime(data['DateTime']).dt.hour*60 + pd.to_datetime(data['DateTime']).dt.minute
    data["doy_sin"], data["doy_cos"] = make_cyclic(data["doy"])
    data["tod_sin"], data["tod_cos"] = make_cyclic(data["tod"])
    
    #data['GPP_prox']= GPP_prox(data)
    #data['WD_sin'], data['WD_cos'] = WD_trans(data)
    data['NEE_NT'] = -data['GPP_NT'] + data['RECO_NT']
    data['NEE_DT'] = -data['GPP_DT'] + data['RECO_DT']

    return data

def custom_filter(data, config, training=True):
    """ Apply certain filters to the data.

    Args:
        data (pd.DataFrame): Dataframe with all the FLUXNET data.
        config (dict): Information about the included variables and the year.

    Returns:
        pd.DataFrame: Filtered data.
    """    
    data['QC'] = quality_check(data, config['nn_1'] + config['nn_1'] + ['TA','SW_IN'])
    
    data = data.replace(-9999, np.nan)
    data = data[~pd.isna(data[config['nn_1'] + config['nn_1'] + ['TA','SW_IN'] + ['NEE']]).any(axis=1)]

    if training:
        mask = (data['Year']==config['year']) & (data['NEE_QC']==0)
    else:
        mask = (data['Year']==config['year'])
    return data[mask]

def unwrap_time(data):
    """
    Takes a TIMESTAMP column of format 20040102 in generates a column for data,
    time, month, year, doy

    Args:
        data (pd.DataFrame): Dataframe with all the FLUXNET data including the time stamp.

    Returns:
        df (pd.DataFrame): Copy of dataframe with standardized date and time columns.
    """

    df = data.copy()

    # Compute DateTime column depending on different formatings of the data
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'DateTime'})
    if 'TIMESTAMP_START' in df.columns:
        # Choose the middle of the timewindow as reference
        df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START']+15, format="%Y%m%d%H%M")
    if all(col in df.columns for col in ['Date', 'Time']):
        df['DateTime'] = df['Date'] + 'T' + df['Time']
    if pd.api.types.is_object_dtype(df['DateTime']):
        df['DateTime'] = pd.to_datetime(df['DateTime'], format = "%Y-%m-%dT%H:%M:%S")
        
    df["Date"] = pd.to_datetime(df['DateTime']).dt.date
    df["Time"] = pd.to_datetime(df['DateTime']).dt.time
    df["Month"] = pd.to_datetime(df['Date']).dt.month
    df["Year"] = pd.to_datetime(df['Date']).dt.year
    df["doy"] = pd.to_datetime(df['DateTime']).dt.dayofyear
    return df


def standardize_column_names(data):
    """
    Changes column names of a dataset according to the relevant_variables file.

    Args:
        data (pd.DataFrame): Dataframe with flux data.

    Returns:
        df (pd.DataFrame): Copy of dataframe with standardized column names.
    """

    df = data.copy()
    for old, new in relevant_variables.mappings.items():
        #if old in data.columns and new not in data.columns:
        if old in df.columns:
            df[new] = df[old]
    return df


def filter_columns_names(data):
    """
    Reduce columns to only the ones considered relevant and specified in 
    relevant_variables file.

    Args:
        data (pd.DataFrame): Dataframe with flux data (ideally the names were 
        previously standardized)

    Returns:
        pd.DataFrame: View of input dataframe with in specified columns
    """
    return data[list(set(data.columns) & set(relevant_variables.variables))]

def wdefcum(data):
    """
    Function to compute cumulative water deficite from precipitation P and latent heat flux LE.
    Equation obtained from Nuno and Markus. This is its simplest form the LE to ET function can be
    made more complex as a next step.

    The names and units of the variables are
        P (unit): precipitation
        LE (unit): latent heat flux
        ET (unit): evapotranspiration
        CWD (unit): cumulative water deficit


    Args:
        data (pd.DataFrame): Dataframe with flux data.


    Returns:
        CWD (float64): cumulative water deficit
    """
    # TODO: Discuss. Is this a proper way to compute it and what does it mean.
    # TODO: Figure out what to do with naming conventions in scientific context.
    P = data['P']
    LE = data['LE']
    if 'P' in data.columns and 'LE' in data.columns:
        n = len(LE)
        # TODO: Figure out the meaning of the magic number here.
        ET = LE / 2.45e6 * 1800
        CWD = np.zeros(n)
        CWD[1:] = np.NaN

        for i in range(1,n):
            CWD[i] = np.minimum(CWD[i-1] + P[i-1] - ET[i-1],0)

        if np.isnan(CWD[i]):
            CWD[i] = CWD[i-1]
    else:
        print('You are missing either P or LE to compute CWD')
        CWD = None
    return CWD


def diffuse_to_direct_rad(data):
    """
    Function to compute a proxy of the ratio between diffuse and direct radiation
    similar to the one of the Tramontana paper.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data.


    Returns:
        SW_ratio (float64): Same dataframe as input with additional SW_ratio column.
    """
    
    # SW_IN is not supposed to be larger than SW_IN_POT
    SW_IN = data['SW_IN'].copy()
    indices = data['SW_IN'] > data['SW_IN_POT']
    SW_IN[indices] = data.loc[indices,'SW_IN_POT']
    
    epsilon = 1e-10  # Prevent division by zero
    SW_ratio = 1 - data['SW_IN']/(data['SW_IN_POT']+epsilon)
    return SW_ratio


def NEE_quality_masks(data):
    """
    Computes quality masks for evaluation depending on availability of measured NEE.
    The criterion are:
        halhourly: NEE is measured
        daily: 50% of the NEE of a day measured
        yearly: half of the days at least 50% measured NEE
        seasonal: half of the days of a season (5 days) have at least 50% measured NEE
        daily_anomalies: pass daily and seasonal quality check
        seasonal_without_yearly: pass seasonal and yearly quality check 

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'NEE_QC' as NEE
        quality flag and columns for site, Year and doy.

    Returns:
        df (pd.DataFrame): Copy of dataframe with all quality masks.
    """    
    df = data.copy()

    # Quality mask per datapoint
    df['QM_halfhourly'] = df['NEE_QC'].apply(lambda x: 0 if x >= 1 else 1)

    # Quality mask per day
    daily_avg = df.groupby(['site', 'Year', 'doy'])['QM_halfhourly'].mean()
    daily_avg = daily_avg.rename('QM_daily')
    df = df.join(daily_avg, on=['site', 'Year', 'doy'])
    df['QM_daily'] = df['QM_daily'].apply(lambda x: 0 if x < 0.5 else 1)

    # Quality mask per year
    yearly_avg = df.groupby(['site', 'Year'])['QM_daily'].mean()
    yearly_avg = yearly_avg.rename('QM_yearly')
    df = df.join(yearly_avg, on=['site', 'Year'])
    df['QM_yearly'] = df['QM_yearly'].apply(lambda x: 0 if x < 0.5 else 1)

    # Quality mask per season
    df['soy'] = df['doy'].apply(lambda x: x//5)
    seasonal_avg = df.groupby(['site', 'Year', 'soy'])['QM_daily'].mean()
    seasonal_avg = seasonal_avg.rename('QM_seasonal')
    df = df.join(seasonal_avg, on=['site', 'Year', 'soy'])  
    df['QM_seasonal'] = df['QM_seasonal'].apply(lambda x: 0 if x < 0.5 else 1)

    # Quality mask for mixed values
    df['QM_daily_anomalies'] = df['QM_daily']*df['QM_seasonal']
    df['QM_seasonal_without_yearly'] = df['QM_seasonal']*df['QM_yearly']
    return df


def daily_means(data, masked=True):
    df = data.copy()

    # Not sure yet
    if not masked:
        indices = True
    else:
        indices = (data['NEE_QC'] == 0)

    for flux in ['GPP', 'RECO', 'NEE']:
        for method in ['DT', 'NT']:
            daily_avg = df.groupby(['site', 'Year', 'doy'])[f'{flux}_{method}'].mean()
            daily_avg = daily_avg.rename(f'{flux}_{method}_daily_avg')
            df = df.join(daily_avg, on=['site', 'Year', 'doy'])
            df[f'{flux}_{method}_rm_daily'] = df[f'{flux}_{method}']\
                                            - df[f'{flux}_{method}_daily_avg']
                                            
    daily_avg = df.groupby(['site', 'Year', 'doy'])['T'].mean()
    daily_avg = daily_avg.rename('T_daily_avg')
    df = df.join(daily_avg, on=['site', 'Year', 'doy'])
    df['T_rm_daily'] = df['T'] - df['T_daily_avg']
                                        
    return df


def quality_check(data, variables, nn_qc=True):
    """
    Introduce yearwise quality_flag on the data following the criterion applied by
    Tramontana.
    The criterion are:
    1. percentage of meteorological gap-filled data is less than 20%
    2. measured NEE covered at least 10% of both daytime and nighttime periods

    Args:
        data (pd.DataFrame): Dataframe with all the flux data.
        variables (list): Variables that serve as input for training
        nn_qc (bool, optional): Checks if the year was filtered out for the nn training. 
        Defaults to True.

    Returns:
        int64: count of unfulfilled criterion. At least one is enough to ignore the year.
    """
    QC = np.zeros(data.shape[0])
    
    for year in data['Year'].unique():
        data_year = data[data['Year'] == year]
        fail_count = 0
        for var in variables:
            if var.endswith('_n') or var.endswith('_s'):
                var = var[:-2]
            if (var+'_QC') in data.columns and var != 'NEE':
                if sum(data_year[var + '_QC'] == 0)/len(data_year) <= 0.8:
                    fail_count += 1
            
        night_data = data_year[data_year['NIGHT'] == 1]
        day_data = data_year[data_year['NIGHT'] == 0]
        
        if len(night_data) == 0 or len(day_data) == 0:
            data.loc[data['Year'] == year,'QC'] = 100
            continue
        
        if sum(night_data['NEE_QC'] == 0)/len(night_data) < 0.1:
            fail_count += 1
        if sum(day_data['NEE_QC'] == 0)/len(day_data) < 0.1:
            fail_count += 1
        QC[data['Year'] == year] = fail_count
        
    return QC


def moving_average(x, w):
    """
    Computes the moving average of window size w over array x

    Args:
        x (float64): array that is convolved over
        w (int64): window size

    Returns:
        float64: moving averages of x
    """
    return np.convolve(x, np.ones(w), 'same') / w


def sw_pot_sm(data):
    """
    Smooth curve of potential incoming radiation computed as 10 day movering averages 
    over SW_IN_POT.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'SW_IN_POT' as column.

    Returns:
        float64: smooth cycle of potential incoming radiation
    """
    SW_POT_sm = moving_average(data['SW_IN_POT'], 480)
    return SW_POT_sm


def sw_pot_sm_diff(data):
    """
    Smooth derivative of the smooth cycle of potential incoming radiation.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'SW_POT_sm' as column.

    Returns:
        SW_POT_sm_diff (float64): smooth derivative of smooth potential incoming radiation
    """

    SW_POT_sm = data['SW_POT_sm']
    SW_POT_sm_diff = np.hstack((np.array(SW_POT_sm[1]-SW_POT_sm[0]),
                                (np.roll(SW_POT_sm,-1) - SW_POT_sm)[1:]))
    SW_POT_sm_diff = moving_average(10000*SW_POT_sm_diff, 480)
    return SW_POT_sm_diff


def WD_trans(data):
    """
    Compute a two dimensional representation of the wind direction.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'WD' as column.

    Returns:
        WD_sin (float64): x dimension of transform
        WD_cos (float64): y dimension of transform
    """
    WD_sin = np.sin(data['WD']/180*np.pi)
    WD_cos = np.cos(data['WD']/180*np.pi)
    return WD_sin, WD_cos

def GPP_prox(data):
    """
    Compute daily GPP average proxy according to Tramontana. 
    NEE_nt_avg represents a RECO avg at night. After scaling its substracted from the daily average.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'WD' as column.

    Returns:
        df (pd.DataFrame): Copy of dataframe with proxies.
    """    
    df = data.copy()
    dt_avg = df[df['NIGHT'] == 0].groupby(['doy', 'Year']).mean()['NEE']
    nt_avg = df[df['NIGHT'] == 1].groupby(['doy', 'Year']).mean()['NEE']
    k = 1-df.groupby(['doy', 'Year']).mean()['NIGHT']
    
    df['NEE_nt_avg'] = df.apply(lambda x: nt_avg[(x['doy'], x['Year'])], axis=1)
    df['NEE_dt_avg'] = df.apply(lambda x: dt_avg[(x['doy'], x['Year'])], axis=1)
    df['k'] = df.apply(lambda x: k[(x['doy'], x['Year'])], axis=1)
    df['GPP_prox'] = (df['NEE_dt_avg'] - df['NEE_nt_avg'])*df['k']
    return df

def normalize(data, var, norm_type='s', masked_normalization=True):
    """
    Normalizes or standardizes a variable from a dataset yearwise.
    For each yearwise transform it applies the same transform for the next years
    for generating variables for the test case. The last year's test set is the
    first year again.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data.
        var (str): variable from the Dataframe that needs to be transformed
        norm_type (str, optional): Marks normalization with 'n' or standardization with 's'.
        Defaults to 's'.
        masked_normalization (bool, optional): Only uses "quality" data for the transform. 
        Defaults to True.

    Returns:
        _type_: _description_
    """    
    df = data.copy()
    
    years = df['Year'].unique()
    years.sort()
    np.append(years, [years[0]])
    
    for i in range(len(years)-1):
        
        if masked_normalization:
            mask = (df['Year']==years[i]) & (df['NEE_QC']==0)
            
        indices_train = (df['Year']==years[i])    
        indices_test = (df['Year']==years[i+1])
        
        if norm_type == 's':
            if masked_normalization:
                df.loc[indices_train, f'{var}_s'] = \
                (df.loc[indices_train, var] - df.loc[mask, var].mean()) / \
                df.loc[mask, var].std()
                
                df.loc[indices_test, f'{var}_s_test'] = \
                (df.loc[indices_test, var] - df.loc[mask, var].mean()) / \
                df.loc[mask, var].std()
            else:
                df.loc[indices_train, f'{var}_s'] = \
                (df.loc[indices_train, var] - df.loc[indices_train, var].mean()) / \
                df.loc[indices_train, var].std()
                
                df.loc[indices_test, f'{var}_s_test'] = \
                (df.loc[indices_test, var] - df.loc[indices_train, var].mean()) / \
                df.loc[indices_train, var].std()
        elif norm_type == 'n':
            if masked_normalization:
                var_max = df.loc[mask, var].abs().max()
                var_min = -var_max
            else:
                var_max = df.loc[indices_train, var].abs().max()
                var_min = -var_max
            df.loc[indices_train, var + f'{var}_n'] = \
                2 * ((df.loc[indices_train, var] - var_min) / (var_max-var_min) -0.5)
            df.loc[indices_test, var + f'{var}_n_test'] = \
                2 * ((df.loc[indices_test, var] - var_min) / (var_max-var_min) -0.5)
    return df


def make_cyclic(x):
    """
    Computes the cyclic representation of a variables.

    Args:
        x (array_like): Input array to be transformed

    Returns:
        (array_like): x axis of transform
        (array_like): y axis of transform
    """
    
    x_norm = 2 * math.pi * x / x.max()
    return np.sin(x_norm), np.cos(x_norm)


def check_available_variables(variables, columns):
    """
    Compares available variables with the desired ones and returns the
    available list.

    Args:
        columns (list): list of dataset columns
        variables (list): list of desired variables (potentially with ending _s, _n)
    
    Returns:
        variables_available (list): list of desired variables that are also in the dataset
    
    """
    
    variables_available = list()
    for var in variables:
        if var.endswith('_n') or var.endswith('_s'):
            suffix = -2
        else:
            suffix = None
        
        if var[:suffix] in columns:
            variables_available.append(var)
        else:
            print(f"Variables {var[:suffix]} not in the dataset.")
            
    return variables_available
