
import os
import time
from argparse import ArgumentParser
import random as orandom
from pathlib import Path
import pandas as pd
import sys
import yaml

#os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 1 7
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".60"

import numpy as np
from nnfluxes.datasets.loaders import load_dataset, BootstrapLoader
from nnfluxes.models.models import BaseModel, Nightconstrained, LUE, Q10, Nightconstrained_Q10, LUE_Q10
from nnfluxes.analysis.postprocessing import merge_data, prepare_data
from nnfluxes.analysis.visualization import monthly_curves, plot_training_curves, taylor_plot

from jax import random
from jax import numpy as jnp
from functools import partial

from sklearn.metrics import r2_score, mean_squared_error


def main(parser: ArgumentParser = None, **kwargs):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("-model", type=str, default="BaseModel", help="Hybrid Model")
    parser.add_argument("-hidden_layer", type=int, default=2, help="batch size")
    parser.add_argument("-hidden_nodes", type=int, default=32, help="batch size")
    parser.add_argument("-nIter", type=int, default=20000, help="training steps")
    parser.add_argument("-drop_p", type=float, default=0, help="Dropout probability")
    parser.add_argument("-weight_decay", type=float, default=5, help="Weight decay")
    #parser.add_argument("-save_curves", type=bool, default=False, help="Should the full curves be saved?")
    parser.add_argument('-TA', action='store_true')
    parser.add_argument('-no-TA', dest='TA', action='store_false')
    parser.add_argument('-SW', action='store_true')
    parser.add_argument('-no-SW', dest='SW', action='store_false')
    parser.add_argument("-year", type=int, default=2006, help="Year of data to use")
    parser.add_argument("-site", type=str, default="DE-Tha", help="Site of data to use")

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    args.TA = int(args.TA)
    args.SW = int(args.SW)

    ################ Define the experiment  ################
    # Model
    model_config = {'hybrid_model':args.model,  # Choose from: BaseModel, NightconstrainedModel, LUE, Q10, LUE_Q10, Nightconstrained_Q10
                    'nn_1': args.hidden_layer*[args.hidden_nodes] + [1],   # hidden layers without input
                    'nn_2': args.hidden_layer*[args.hidden_nodes] + [1],
                    'p': args.drop_p*0.1,
                    'ensemble_size': 100,
                    'weight_decay': args.weight_decay*0.1,
    }
    
    data_config = {'hybrid_model':model_config['hybrid_model'],
                    'nn_1': ['VPD', 'doy_cos', 'doy_sin'] + args.SW*['SW_IN'], #'PA', 'RH', 'WS', 'P', 'tod_sin', 'tod_cos' #'SW_IN_POT', 'TA', 'WS', 'SWC_1', 
                    'nn_2': ['VPD', 'doy_cos', 'doy_sin'] + args.TA*['TA'], #'PA', 'RH', 'WS', 'P', 'tod_sin', 'tod_cos' #'SWC_1', 'TS_1',
                    'split': 'random',
                    'site': args.site,
                    'year': args.year,
                    'filter': 'Tramontana',
                    'target': 'NEE',
    }
    
    trainer_config = { 'batch_size': 256,
                        'nIter': args.nIter}
    
    ################ Save the experiment setup  ################    
    experiment_dict = {'model_config': model_config, 'data_config': data_config, 'trainer_config': trainer_config}

    
    results_path = Path(__file__).parent.parent.joinpath("results",f"{args.model}_{args.hidden_layer}x{args.hidden_nodes}_p{args.drop_p}_wd{args.weight_decay}_{args.site}_{args.year}")
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path.joinpath('experiment_dict.yaml'), 'w') as file:
        yaml.dump(experiment_dict, file)
    
    # Add number of inputs layers of the model config
    model_config['nn_1'] = [len(data_config['nn_1'])] + model_config['nn_1']
    model_config['nn_2'] = [len(data_config['nn_2'])] + model_config['nn_2']
    
    ################ Load the data  ################
    #ToDO: Define one loading function for training and one for inference
    train, val, test, out, all = load_dataset(data_config, training=True)
    EV1_train, driver1_train, EV2_train, driver2_train, NEE_train, \
    EV1_val, driver1_val, EV2_val, driver2_val, NEE_val, \
    EV1_test, driver1_test, EV2_test, driver2_test, NEE_test, NEE_max_abs = out

    EV1, driver1, EV2, driver2, NEE, data  = all

    data_train = [EV1_train, EV2_train, driver1_train, driver2_train, NEE_train]        
    data_val = [EV1_val, EV2_val, driver1_val, driver2_val, NEE_val]
    data_test = [EV1_test, EV2_test, driver1_test, driver2_test, NEE_test]

    rng_key = random.PRNGKey(1)
    rng_key, rng_key1, rng_key2 = random.split(rng_key, 3)

    dataset = BootstrapLoader(EV1_train, EV2_train, driver1_train, driver2_train, NEE_train, 
                            NEE_max_abs=NEE_max_abs, batch_size=trainer_config['batch_size'], ensemble_size=model_config['ensemble_size'], 
                            rng_key=rng_key1)

    regressor = getattr(sys.modules[__name__], model_config['hybrid_model'])
    del model_config['hybrid_model']
    regressor = regressor(**model_config)
    
    ################ Train the model  ################
    regressor.fit(dataset, data_train, data_val, data_test, nIter=args.nIter, rng_key=rng_key2)

    ################ Save the training infos  ################
    for log, name in zip([regressor.train_log, regressor.val_log, regressor.test_log, regressor.Q10_log], ['train', 'val', 'test', 'Q10']):
        log_df = pd.DataFrame(data=np.transpose(np.stack(log)))
        csv_name = f"{name}_log.csv"
        results_file = results_path.joinpath(csv_name)
        log_df.to_csv(results_file)
        
    train_log = pd.read_csv(results_path.joinpath('train_log.csv'), index_col=0)
    val_log = pd.read_csv(results_path.joinpath('val_log.csv'), index_col=0)
    test_log = pd.read_csv(results_path.joinpath('test_log.csv'), index_col=0)
    Q10_log = pd.read_csv(results_path.joinpath('Q10_log.csv'), index_col=0)
    
    lr1 = [regressor.schedule1(i) for i in range(0,args.nIter,100)]
    lr2 = [regressor.schedule2(i) for i in range(0,args.nIter,100)]
    if hasattr(regressor, 'schedule_Q10'):
        lr_Q10 = [regressor.schedule_Q10(i) for i in range(0,args.nIter,100)]
    else: 
        lr_Q10 = [1 for i in range(0,args.nIter,100)]

    # Store the learning rate in csv
    lr_df = pd.DataFrame(data=np.transpose(np.stack([lr1, lr2, lr_Q10])))
    csv_name = f"lr_logs.csv"
    results_file = results_path.joinpath(csv_name)
    lr_df.to_csv(results_file)

    plot_training_curves(train_log, val_log, test_log, Q10_log, lr_df, results_path, moving_window_size=20, format='single')
    plot_training_curves(train_log, val_log, test_log, Q10_log, lr_df, results_path, moving_window_size=20, format='aggregated')
    plot_training_curves(train_log, val_log, test_log, Q10_log, lr_df, results_path, moving_window_size=20, format='dataset_wise')

    ################ Inference  ################
    output_train = list(regressor.posterior(EV1[data['split'].isin([1])], EV2[data['split'].isin([1])], driver1[data['split'].isin([1])], driver2[data['split'].isin([1])]))
    output_val = list(regressor.posterior(EV1[data['split'].isin([2])], EV2[data['split'].isin([2])], driver1[data['split'].isin([2])], driver2[data['split'].isin([2])]))
    output_test = list(regressor.posterior(EV1[data['split'].isin([3])], EV2[data['split'].isin([3])], driver1[data['split'].isin([3])], driver2[data['split'].isin([3])]))
    output_all = list(regressor.posterior(EV1, EV2, driver1, driver2))


    for output in [output_train, output_val, output_test, output_all]:
        for i in range(3):
            output[i]= np.concatenate([output[i], output[i].mean(axis=0)[None,:,:]], axis=0)
        
    ################ Analysis  ################
    # Create the results_oath folder if it does not exist
    #ToDo: Make the plot with the log curves for training, validation and test (potentially also Q10 and LR and mark best model)
    
    results = pd.DataFrame()
    
    for i in range(model_config['ensemble_size']+1):
        row = dict()
        for measure_str, measure in iter(zip(['rmse', 'r2'],[partial(mean_squared_error, squared=False), r2_score])):
            for method in ['DT', 'NT']:
                for split_str, split, subset in iter(zip(['train', 'val', 'test' ,'all'], [output_train, output_val, output_test, output_all], [[1], [2], [3], [1,2,3]])):
                    for j, flux in enumerate(['NEE', 'GPP', 'RECO']):
                        row[f'{measure_str}_{flux}_{method}_{split_str}'] = measure(data[data['split'].isin(subset)][f'{flux}_{method}'], split[j][i,:])
        results = pd.concat([results, pd.DataFrame(row, index=[i])], axis=0)

    #ToDo: Make the plot with the dimensions for each of the ensemble members and for the mean
    
    csv_name = "performance.csv"
    results_file = results_path.joinpath(csv_name)
    results.to_csv(results_file)

    # Delete all the unused data.
    del output_train, output_val, output_test, output_all, output, train, val, test, out, all
    
    all = load_dataset(data_config, training=False)
    EV1, driver1, EV2, driver2, NEE, data  = all
    output_all = list(regressor.posterior(EV1, EV2, driver1, driver2))
        
    for i, flux_str in enumerate(['NEE', 'GPP', 'RECO']):
        flux = pd.DataFrame(data=np.transpose(output_all[i][:,:,0]))
        flux.index = data.index
        flux[f'{flux_str}_DT'] = data[f'{flux_str}_DT']
        flux[f'{flux_str}_NT'] = data[f'{flux_str}_NT']
        flux['NEE_QC'] = data['NEE_QC']
        csv_name = f'{flux_str}.csv'
        results_file = results_path.joinpath(csv_name)
        flux.to_csv(results_file)
    if output_all[3] is not None:
        flux = pd.DataFrame(data=np.transpose(output_all[3][:,:,0]))
        flux.index = data.index
        flux['NEE_QC'] = data['NEE_QC']
        csv_name = 'LUE.csv'
        results_file = results_path.joinpath(csv_name)
        flux.to_csv(results_file)
        
    GPP = pd.read_csv(results_path.joinpath('GPP.csv'), index_col=0)
    RECO = pd.read_csv(results_path.joinpath('RECO.csv'), index_col=0)
    NEE = pd.read_csv(results_path.joinpath('NEE.csv'), index_col=0)
    
    if output_all[3] is not None:
        LUE = pd.read_csv(results_path.joinpath('LUE.csv'), index_col=0)
    else:
        LUE = None
    
    GPP, RECO, NEE = merge_data(GPP, RECO, NEE, site_name=data_config["site"], year=data_config["year"])
    GPP, RECO, NEE, LUE = prepare_data(GPP, RECO, NEE, LUE, ensemble_size=model_config['ensemble_size'])
    
    #args.save_curves
    monthly_curves('NEE', NEE, results_path=results_path)
    monthly_curves('GPP', GPP, results_path=results_path)
    monthly_curves('RECO', RECO, results_path=results_path)
    if output_all[3] is not None:
        monthly_curves('LUE', LUE, compare_to=[], results_path=results_path)

    taylor_plot(NEE, GPP, RECO, ensemble_size=model_config['ensemble_size'], filtered=True, results_path=results_path)

    #Todo: diurnal cycles plot
    #Todo: Potentially save the mean and std fluxes.

if __name__ == '__main__':
    main()