
import pandas as pd
from pathlib import Path
from nnfluxes.analysis.utils import unfold_time

def merge_data(GPP, RECO, NEE, site_name, year):
    # Add ANN partitioning
    #site_name = 'DE-Tha'
    file = f'{site_name}_ANN_partitioning.csv'
    data_folder = Path(__file__).parent.parent.parent.parent.joinpath('dml-4-fluxes','data')
    data_ann = pd.read_csv(data_folder.joinpath('ANNpartitioning',file))

    GPP['GPP_ann'] = data_ann[data_ann['Year'] == year]['GPPann'].values
    RECO['RECO_ann'] = data_ann[data_ann['Year'] == year]['RECOann'].values
    NEE['NEE_ann'] = (-data_ann[data_ann['Year'] == year]['GPPann'] + data_ann[data_ann['Year'] == year]['RECOann']).values

    GPP['MeasurementNEE_mask'] = data_ann['MeasurementNEE_mask']
    RECO['MeasurementNEE_mask'] = data_ann['MeasurementNEE_mask']
    NEE['MeasurementNEE_mask'] = data_ann['MeasurementNEE_mask']


    GPP = GPP.fillna(-9999.0)
    RECO = RECO.fillna(-9999.0)
    NEE = NEE.fillna(-9999.0)
    
    # Add DML partitioning
    data_folder = Path(__file__).parent.parent.parent.parent.joinpath('dml-4-fluxes','data', 'Fluxnet-2015','DMLPartitioning', 'FPDML_all_2023-04-15_2', f'FLX_{site_name}')

    partition = pd.read_csv(data_folder.joinpath('orth_partitioning.csv'))
    partition['DateTime'] = pd.to_datetime(partition['DateTime'], format = "%Y-%m-%d %H:%M:%S")
    partition = partition.set_index('DateTime')

    GPP['GPP_orth'] = partition[partition.index.year == year]['GPP_orth'].values
    RECO['RECO_orth'] = partition[partition.index.year == year]['RECO_orth'].values
    NEE['NEE_orth'] = partition[partition.index.year == year]['NEE_orth'].values
    
    GPP = GPP.fillna(-9999.0)
    RECO = RECO.fillna(-9999.0)
    NEE = NEE.fillna(-9999.0)
    
    
    return GPP, RECO, NEE

def prepare_data(GPP, RECO, NEE, LUE=None, ensemble_size=100):
    GPP = unfold_time(GPP)
    RECO = unfold_time(RECO)
    NEE = unfold_time(NEE)
    
    GPP['GPP_mean'] = GPP.iloc[:,0:ensemble_size].values.mean(axis=1)
    GPP['GPP_std'] = GPP.iloc[:,0:ensemble_size].values.std(axis=1)

    RECO['RECO_mean'] = RECO.iloc[:,0:ensemble_size].values.mean(axis=1)
    RECO['RECO_std'] = RECO.iloc[:,0:ensemble_size].values.std(axis=1)

    NEE['NEE_mean'] = NEE.iloc[:,0:ensemble_size].values.mean(axis=1)
    NEE['NEE_std'] = NEE.iloc[:,0:ensemble_size].values.std(axis=1)
    
    # transform all columns that are float64 to float32
    GPP[GPP.columns[GPP.dtypes == 'float64']] = GPP[GPP.columns[GPP.dtypes == 'float64']].astype('float32')
    RECO[RECO.columns[RECO.dtypes == 'float64']] = RECO[RECO.columns[RECO.dtypes == 'float64']].astype('float32')
    NEE[NEE.columns[NEE.dtypes == 'float64']] = NEE[NEE.columns[NEE.dtypes == 'float64']].astype('float32')
    
    if LUE is not None:
        LUE = unfold_time(LUE)
        LUE['LUE_mean'] = LUE.iloc[:,0:ensemble_size].values.mean(axis=1)
        LUE['LUE_std'] = LUE.iloc[:,0:ensemble_size].values.std(axis=1) 
        LUE[LUE.columns[LUE.dtypes == 'float64']] = LUE[LUE.columns[LUE.dtypes == 'float64']].astype('float32')
    
    return GPP, RECO, NEE, LUE