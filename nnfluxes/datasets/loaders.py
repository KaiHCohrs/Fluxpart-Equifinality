import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from nnfluxes.datasets.preprocessing import preprocess, custom_filter
#import tensorflow as tf
from torch.utils import data

from jax import grad, vmap, random, jit
from jax import numpy as np
import numpy as onp
from matplotlib import pyplot as plt

from functools import partial
import itertools
data_dir = Path("data")


#Todo normalize names
def impose_noise(data_nn, SIFnoise_std=0.1, NEEnoise_std=0.08):
    """
    add noise to NEE
    NEEnoise_std: we add a heteroscedastic noise that scales for NEE magnitude (8% of the NEE magnitude)
    """
    std_NEE = NEEnoise_std * data_nn.NEE_canopy.abs()

    data_night = data_nn.loc[data_nn.APAR_label == 0, :].copy()
    data_day = data_nn.loc[data_nn.APAR_label == 1, :].copy()

    np.random.seed(42)
    noise_NEE = np.random.normal(0, std_NEE)

    # add SIF noise
    # add NEE noise
    data_nn.loc[:, 'NEE_obs'] = data_nn['NEE_canopy'] + noise_NEE

    return data_nn, data_day


def standard_x(x_train, x_test=None):
    # the mean and std values are only calculated by training set
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train1 = ((x_train - x_mean) / x_std).values
    if x_test is not None:
        x_test1 = ((x_test - x_mean) / x_std).values
        return x_train1, x_test1
    return x_train1

#Todo change the function such that extracts data from a specific fluxnet site and year
#Todo add the option to use the DT data
#Todo add an input for choosing the EVs
#Todo add an input for choosing the hybrid model
def load_dataset(config, training=True):
    data_dir = Path(__file__).parent.parent.parent.parent.joinpath('dml-4-fluxes', 'data', 'Fluxnet-2015', f'FLX_{config["site"]}')
    for file in data_dir.glob('*'):
        # Most sites have half-hourly data
        if file.name.startswith(f'FLX_{config["site"]}_FLUXNET2015_FULLSET_HH'):
            filename = file.name
        # A few have hourly data
        elif file.name.startswith(f'FLX_{config["site"]}_FLUXNET2015_FULLSET_HR'):
            filename = file.name
        
    data = pd.read_csv(data_dir / filename)
    data['site'] = config['site']

    # Apply whole preprocessing pipeline
    data = preprocess(data)
    data_filtered = custom_filter(data, config, training=training)
    data_filtered = data_filtered.astype({col: 'float32' for col in data_filtered.select_dtypes('float64').columns})
    
    # add noise to NEE & SIF simulations
    # data_nn, data_day = impose_noise(data_nn)

    if config['split'] == 'random':
        train, test = train_test_split(data_filtered, test_size=0.3, random_state=31, shuffle=True)
        val, test = train_test_split(test, test_size=0.3, random_state=31, shuffle=True)

    data_filtered.loc[train.index,'split'] = 1
    data_filtered.loc[val.index,'split'] = 2
    data_filtered.loc[test.index,'split'] = 3

    EV1_label = config['nn_1']
    EV2_label = config['nn_2']
    NEE_label = config['target']
    
    EV1_train = train[EV1_label].astype('float32')  # EV for GPP
    EV2_train = train[EV2_label].astype('float32')  # EV for Reco
    NEE_train = train[NEE_label].values[:,None]
    
    EV1_val = val[EV1_label].astype('float32')  # EV for GPP
    EV2_val = val[EV2_label].astype('float32')  # EV for Reco
    NEE_val = val[NEE_label].values[:,None]

    EV1_test = test[EV1_label].astype('float32')  # EV for GPP
    EV2_test = test[EV2_label].astype('float32')  # EV for Reco
    NEE_test = test[NEE_label].values[:,None]

    EV1_train, EV1_val, EV1_test = EV1_train.values, EV1_val.values, EV1_test.values
    EV2_train, EV2_val, EV2_test = EV2_train.values, EV2_val.values, EV2_test.values


    # Drivers for the hybrid model
    if config['hybrid_model'] == 'Nightconstrained_Q10':
        driver1_label = ['NIGHT']
    elif config['hybrid_model'] == 'Nightconstrained':
        driver1_label = ['NIGHT']
    else:
        driver1_label = ['SW_IN']
    driver2_label = ['TA']
    
    if config['hybrid_model'] == 'PriorBaseModel':
        driver1_label = ['GPP_DT']
        driver2_label = ['RECO_DT']

    driver1_train = train[driver1_label].astype('float32')  # driver for GPP
    driver2_train = train[driver2_label].astype('float32')  # driver for Reco
    
    driver1_val = val[driver1_label].astype('float32')  # driver for GPP
    driver2_val = val[driver2_label].astype('float32')  # driver for Reco
    
    driver1_test = test[driver1_label].astype('float32')  # driver for GPP
    driver2_test = test[driver2_label].astype('float32')  # driver for Reco
    
    driver1_train, driver1_val, driver1_test = driver1_train.values, driver1_val.values, driver1_test.values
    driver2_train, driver2_val, driver2_test = driver2_train.values, driver2_val.values, driver2_test.values

    # EVs and drivers on the whole dataset
    EV1 = data_filtered[EV1_label].astype('float32').values
    driver1 = data_filtered[driver1_label].astype('float32').values
    EV2 = data_filtered[EV2_label].astype('float32').values
    driver2 = data_filtered[driver2_label].astype('float32').values
    NEE = data_filtered[NEE_label].astype('float32').values[:,None]

    # Y_data Normalization
    NEE_max_abs = (np.abs(NEE_train)).max()

    out = [EV1_train, driver1_train, EV2_train, driver2_train, NEE_train,
            EV1_val, driver1_val, EV2_val, driver2_val, NEE_val,
            EV1_test, driver1_test, EV2_test, driver2_test, NEE_test, NEE_max_abs]
    
    # Convert columns with float values to float32

    all = [EV1, driver1, EV2, driver2, NEE] + [data_filtered[['GPP_DT', 'GPP_NT',
                                                            'RECO_DT', 'RECO_NT',
                                                            'NEE_DT', 'NEE_NT', 'NEE_QC', 'split']]]

    if training:
        return train, val, test, out, all
    else:
        return all

class BootstrapLoader(data.Dataset):
    def __init__(self, EV1, EV2, driver1, driver2, NEE, NEE_max_abs=1, batch_size=128, ensemble_size=32, fraction=0.5, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.N = EV1.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.bootstrap_size = int(self.N*fraction)
        self.key = rng_key
        self.NEE_max_abs = NEE_max_abs
        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        self.EV1, self.EV2, self.driver1, self.driver2, self.NEE = vmap(self.__bootstrap, (None,None,None,None,None,0))(EV1, EV2, driver1, driver2, NEE, keys)
        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0,0,0,0,0))(self.EV1, self.EV2, self.driver1, self.driver2, self.NEE)

    #@partial(jit, static_argnums=(0,)) <- SHould be jittable but might not bring much advantage
    def normalization_constants(self, EV1, EV2, driver1, driver2, NEE):
        mu_EV1, sigma_EV1 = EV1.mean(0), EV1.std(0)
        mu_EV2, sigma_EV2 = EV2.mean(0), EV2.std(0)
        mu_driver1, sigma_driver1 = np.zeros(driver1.shape[1],), driver1.max(0)
        mu_driver2, sigma_driver2 = np.zeros(driver2.shape[1],), driver2.max(0)
        
        mu_NEE, sigma_NEE = np.zeros(NEE.shape[1],), self.NEE_max_abs*np.ones(NEE.shape[1],) # np.abs(NEE).max(0)
        
        return (mu_EV1, sigma_EV1), (mu_EV2, sigma_EV2), (mu_driver1, sigma_driver1),  (mu_driver2, sigma_driver2), (mu_NEE, sigma_NEE)

    @partial(jit, static_argnums=(0,))
    def __bootstrap(self, EV1, EV2, driver1, driver2, NEE, key):
        idx = random.choice(key, self.N, (self.bootstrap_size,), replace=False)
        inputs1 = EV1[idx,:]
        inputs2 = EV2[idx,:]
        inputs3 = driver1[idx,:]
        inputs4 = driver2[idx,:]
        targets = NEE[idx,:]
        return inputs1, inputs2, inputs3, inputs4, targets

    @partial(jit, static_argnums=(0,)) # Looks good, everything seems local
    def __data_generation(self, key, EV1, EV2, driver1, driver2, NEE, norm_const):
        'Generates data containing batch_size samples'
        (mu_EV1, sigma_EV1), (mu_EV2, sigma_EV2), (mu_driver1, sigma_driver1), (mu_driver2, sigma_driver2), (mu_NEE, sigma_NEE) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        EV1 = EV1[idx,:]
        EV2 = EV2[idx,:]
        driver1 = driver1[idx,:]
        driver2 = driver2[idx,:]
        NEE = NEE[idx,:]
        EV1 = (EV1 - mu_EV1)/sigma_EV1
        EV2 = (EV2 - mu_EV2)/sigma_EV2
        driver1 = (driver1 - mu_driver1)/sigma_driver1
        driver2 = (driver2 - mu_driver2)/sigma_driver2
        NEE = (NEE - mu_NEE)/sigma_NEE
        return EV1, EV2, driver1, driver2, NEE

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs1, inputs2, inputs3, inputs4, targets = vmap(self.__data_generation, (0,0,0,0,0,0,0))(keys, 
                                                                                        self.EV1, 
                                                                                        self.EV2, 
                                                                                        self.driver1, 
                                                                                        self.driver2, 
                                                                                        self.NEE, 
                                                                                        self.norm_const)
        return inputs1, inputs2, inputs3, inputs4, targets