""" Functions for generating synthetic datasets """
import os
import random

import pandas as pd
import numpy as np
from pathlib import Path

import dml4fluxes.datasets.relevant_variables as relevant_variables
from ..experiments.utility import get_available_sites


def synthetic_dataset(data, Q10, relnoise, version='simple', pre_computed=False, site_name=None):
    """
    Generates or loads a precomputed dataset based on the Q10 model for RECO and LUE model for GPP.
    In its simplest form it resembles the model from the book chapter.
    
    Args:
        data (pd.DataFrame): FLUXNET dataset
        Q10 (float): Q10 specifying the value of the Q10 model based simulation
        relnoise (float): non-negative sd of noise applied as a factor (1+noise) to the final NEE
        version (str, optional): Specifies if we use the simple bookchapter
        functions or a more complex models for generated data. Defaults to 'simple'.
        pre_computed (bool, optional): If True takes already computed dataset
        with a certain noise level so that the exact same data can be reused 
        for the standard partitioning methods. Defaults to FALSE.
        site (str, optional): Site the precomputed data should come from. Defaults to None.

    Returns:
        data (pd.DataFrame): Dataset with additional columns with intermediate values of the data
        generation process. In particular RECO_syn, GPP_syn, NEE_syn_clean (noise free), NEE_syn
    """
    if pre_computed:
        data_syn = pd.read_csv(Path(__file__)\
                    .parent.parent.parent.joinpath('data','Fluxnet-2015','generated_data',
                                                f'{site_name}_{version}_{str(Q10)}.csv'))
        
        # Unfortunatelly pre_computed sets were stored without proper indexing. Take indices form dataset
        data_syn.index = data.index
        
        if 'Unnamed: 0' in data_syn.columns:
            data_syn = data_syn.drop('Unnamed: 0', axis=1)

        data = pd.merge(data, data_syn[[f'GPP_{version}_{str(Q10)}',
                                        f'RECO_{version}_{str(Q10)}',
                                        f'NEE_{version}_{str(Q10)}_clean',
                                        f'NEE_{version}_{str(Q10)}_{str(relnoise)}']],
                                        how='left', right_index=True, left_index=True)
        
        data = data.rename(columns={f'GPP_{version}_{str(Q10)}': 'GPP_syn',
                            f'RECO_{version}_{str(Q10)}': 'RECO_syn',
                            f'NEE_{version}_{str(Q10)}_clean': 'NEE_syn_clean',
                            f'NEE_{version}_{str(Q10)}_{str(relnoise)}': 'NEE_syn',
                            })

    else:
        if 'SW_POT_sm' not in data.columns:
            data = sw_pot_sm(data)
            
        if 'SW_POT_sm_diff' not in data.columns:
            data = sw_pot_sm_diff(data)
            
        SW_IN = data['SW_IN']
        SW_POT_sm = data['SW_POT_sm']
        SW_POT_sm_diff = data['SW_POT_sm_diff']
        TA = data['TA']
        VPD = data['VPD']
        
        #if version=='simple':
        RUE_syn = 0.5 * np.exp(-(0.1*(TA-20))**2) * np.minimum(1, np.exp(-0.1 * (VPD-10)))
        if version == 'medium1.0':
            kw = -0.11
            TSopt = 15
            kTS = 7
            WI = 35
        
            if 'SWC_1' and 'SWC_2' in data.keys():
                SWC =  3/4 * data['SWC_1'] + 1/4 * data['SWC_2']
                fW = 1/(1+np.exp(kw * (SWC - WI))),
            else:
                fW = 1

            if 'TS_1' and 'TS_2' in data.keys():
                TS =  3/4 * data['TS_1'] + 1/4 * data['TS_2']
                fTS = (2 * np.exp(-(TS - TSopt)/kTS))/(1 + np.exp((-(TS - TSopt)/kTS)**2))
            else:
                fTS = 1
            
            fVPD = np.exp(-0.1 * (VPD-10))
            fTA = ((np.exp(-(0.1*(TA-20))**2)))
            #fTA = ((TA - TA.min())*(TA - TA.max()))/((TA - TA.min())*(TA - TA.max())-(TA - 20)**2)
            RUE_syn = 0.5 * fTA * fTS * np.minimum(fVPD, fW).squeeze()
        
        GPP_syn = RUE_syn * SW_IN / 12.011
        Rb_syn = SW_POT_sm * 0.01 - SW_POT_sm_diff * 0.005
        Rb_syn = 0.75 * (Rb_syn - np.nanmin(Rb_syn) + 0.1*np.pi)
        RECO_syn = Rb_syn * Q10 ** (0.1*(TA-15.0))
        NEE_syn_clean = RECO_syn - GPP_syn 
        NEE_syn = NEE_syn_clean * (1 + relnoise * np.random.normal(size=NEE_syn_clean.size))
        
        data['RUE_syn'] = RUE_syn
        data['GPP_syn'] = GPP_syn
        data['Rb_syn'] = Rb_syn
        data['RECO_syn'] = RECO_syn
        data['NEE_syn_clean'] = NEE_syn_clean
        data['NEE_syn'] = NEE_syn
        
    return data

def generate_data(data, rel_noise_levels=[0], Q10=1.5, version='simple', seed=1):
    random.seed(seed)
    np.random.seed(seed)
    gen_columns = list()
    if ('SWC_1' and 'SWC_2' not in data.keys()) and ('TS_1' and 'TS_2' not in data.keys()):
        data['simple'] = 1
    else: data['simple'] = 0
    
    for rel_noise in rel_noise_levels:
        data = synthetic_dataset(data, Q10, rel_noise, version)
        data[f'NEE_{version}_{str(Q10)}_clean'] = data['NEE_syn_clean']
        data[f'NEE_{version}_{str(Q10)}_{str(rel_noise)}'] = data['NEE_syn']
        data[f'RECO_{version}_{str(Q10)}'] = data['RECO_syn']
        data[f'GPP_{version}_{str(Q10)}'] = data['GPP_syn']
        gen_columns.append(f'NEE_{version}_{str(Q10)}_{str(rel_noise)}')
    gen_columns.append(f'NEE_{version}_{str(Q10)}_clean')
    gen_columns.append(f'RECO_{version}_{str(Q10)}')
    gen_columns.append(f'GPP_{version}_{str(Q10)}')
    gen_columns.append('simple')
        
    return data, gen_columns


def generate_all_data(rel_noise_levels=[0], Q10=1.5, version='simple', seed=1, year=2015):
    sites = get_available_sites(f"/usr/users/kcohrs/bayesian-q10/data/Fluxnet-{year}")
    random.seed(seed)
    np.random.seed(seed)
    
    site_seeds = np.random.randint(1,100000, len(sites))
    for i, site in enumerate(sites):
        print(f'Seed used: {site_seeds[i]}')
        data = load_data(site, year=2015, add_ANN=False)
        data = standardize_column_names(data)
        data, gen_columns = generate_data(data, rel_noise_levels, Q10, version, seed=site_seeds[i])
        
        data[gen_columns].to_csv(f"/usr/users/kcohrs/bayesian-q10/data/Fluxnet-{year}/generated_data/{site}_{version}_{str(Q10)}.csv", index=False)
        print(f'Site {site} finished.')


def prepare_std_flx_part(site, year=2015, version='simple', Q10=1.5):
    data = load_data(site, year=2015, add_ANN=False)
    
    gen_NEE = pd.read_csv(f"/usr/users/kcohrs/bayesian-q10/data/Fluxnet-{year}/generated_data/{site}_{version}_{str(Q10)}.csv")
    if 'Unnamed: 0' in gen_NEE.columns:
        gen_NEE = gen_NEE.drop('Unnamed: 0', axis=1)
    
    data = pd.merge(data, gen_NEE, how='left', right_index=True, left_index=True)
    data = unwrap_time(data)
    data = standardize_column_names(data)
    
    data_new = data[['Year', 'TA', 'doy', 'Time', 'VPD', 'SW_IN', 'simple', f'GPP_{version}_{str(Q10)}', f'RECO_{version}_{str(Q10)}']+[col for col in data.columns if col.startswith(f'NEE_{version}')]].copy()
    #data.drop('Unnamed: 0', axis=1, inplace=True)
    #data.drop(['VPD_n', 'SW_ratio', 'RECO_NT', 'RECO_DT', 'RECO_orth_res', 'code', 'GPP_NT', 'RECO_orth', 'site', 'Month', 'TA_n', 'wdefcum', 'LUE_orth', 'GPP_DT', 'NEE_orth', 'NIGHT'], axis=1, inplace=True)
    #data.drop('GPP_orth', axis=1, inplace=True)
    data_new['Hour'] = data_new.apply(lambda x: time_to_dec(x['Time']), axis=1)
    data_new.drop('Time', axis=1, inplace=True)
    data_new.rename(columns={'doy':'DoY', 'TA': 'Tair', 'SW_IN':'Rg'}, inplace=True)
    data_new['Ustar'] = np.random.normal(1, 0.01, len(data_new))
    data_new.to_csv(f"/usr/users/kcohrs/bayesian-q10/data/Fluxnet-{year}/generated_data/{site}_{version}_{str(Q10)}.csv")


def prepare_std_flux_all_data(year=2015, version='simple', Q10=1.5):
    sites = get_available_sites(f"/usr/users/kcohrs/bayesian-q10/data/Fluxnet-{year}")
    for site in sites:
        prepare_std_flx_part(site, year=2015, version=version, Q10=Q10)
        print(f'Site {site} finished.')