
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product

def plot_training_curves(train_log, val_log, test_log, Q10_log, lr_df, results_path=None, moving_window_size=20, format='single'):
    if format == 'single':
        fig, axes = plt.subplots(10,10, figsize= (50,50), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()):
            curve_train = train_log.iloc[i,:]
            curve_val = val_log.iloc[i,:]
            curve_test = test_log.iloc[i,:]
            curve_Q10 = Q10_log.iloc[i,:]
            
            ax.plot(curve_train, color="blue", label="train", alpha=0.5)
            ax.plot(curve_val, color="red", label="val", alpha=0.5)
            ax.plot(curve_test, color="green", label="test", alpha=0.5)
            
            # Mark the minimum of each curve with a dotted vertical line
            ax.axvline(x=curve_train.idxmin(), color="blue", linestyle="dotted")
            ax.axvline(x=curve_val.idxmin(), color="red", linestyle="--")
            ax.axvline(x=curve_test.idxmin(), color="green", linestyle="dotted")
            
            # Moving average of values
            curve_train = curve_train.rolling(window=moving_window_size).mean()
            curve_val = curve_val.rolling(window=moving_window_size).mean()
            curve_test = curve_test.rolling(window=moving_window_size).mean()
            
            ax.plot(curve_train, color="blue", label="train_mv_avg")
            ax.plot(curve_val, color="red", label="val_mv_avg")
            ax.plot(curve_test, color="green", label="test_mv_avg")
            
            # Plot the Q10 curve with its own y-axis
            ax_Q10 = ax.twinx()
            ax_Q10.plot(curve_Q10, color="black", label="Q10")
            ax_Q10.set_ylabel("Q10", color="black")
            ax_Q10.tick_params(axis='y', labelcolor="black")
            
            # Make an x tick every 20 epochs
            ax.set_xticks(range(0, len(curve_train), 50))
            # Make the x-ticks labels be the number of the epoch
            ax.set_xticklabels(range(0, len(curve_train)* 100, 5000) )
            ax.set_yscale('log')
            # add legend to plot
            ax.legend()
            ax.set_title(f"Curve {i}")
        fig.suptitle("Training curves")
        if results_path:
            fig.savefig(results_path.joinpath('training_curves.png'))
    elif format == 'aggregated':
        fig, axes = plt.subplots(3,1, figsize= (10,10))
        
        curve_train = train_log.mean(axis=0)
        curve_val = val_log.mean(axis=0)
        curve_test = test_log.mean(axis=0)
        
        # Moving average of values
        curve_train = curve_train.rolling(window=moving_window_size).mean()
        curve_val = curve_val.rolling(window=moving_window_size).mean()
        curve_test = curve_test.rolling(window=moving_window_size).mean()
            
        axes[0].plot(curve_train, color="blue", label="train")
        axes[0].plot(curve_val, color="red", label="val")
        axes[0].plot(curve_test, color="green", label="test")

        # Make an x tick every 20 epochs
        axes[0].set_xticks(range(0, len(curve_train), 50))
        # Make the x-ticks labels be the number of the epoch
        axes[0].set_xticklabels(range(0, len(curve_train)* 100, 5000) )
        axes[0].set_yscale('log')
        # add legend to plot
        axes[0].legend()
        
        # for pandas dataframe lr_df plot all its columns as timeseries in axes[1]
        lr_df.plot(ax=axes[1])
        axes[1].set_title("Learning rate")
        axes[1].set_xticks(range(0, len(lr_df), 50))
        axes[1].set_xticklabels(range(0, len(lr_df)* 100, 5000) )
        axes[1].set_yscale('log')
        axes[1].legend()

        # for pandas dataframe Q10_log plot all its rows as timeseries in axes[2]
        Q10_log.T.plot(ax=axes[2])
        axes[2].set_title("Q10")
        axes[2].set_xticks(range(0, len(Q10_log.T), 50))
        axes[2].set_xticklabels(range(0, len(Q10_log.T)* 100, 5000) )
        axes[2].get_legend().remove()
        
        fig.suptitle("Training curves")
        if results_path:
            fig.savefig(results_path.joinpath('training_curves_aggregated.png'))
    elif format == 'dataset_wise':
        fig, axes = plt.subplots(2,3, figsize= (15,10))

        sets = [train_log, val_log, test_log]
        labels = ['train', 'val', 'test']
        colors = ['blue', 'red', 'green']



        for i, ax in enumerate(axes[0,:].flatten()):
            for j in range(sets[i].shape[0]):
                curve = sets[i].iloc[j,:]
                
                curve = curve.rolling(window=moving_window_size).mean()
                ax.plot(curve, color=colors[i], label=labels[i], alpha=0.2)            

            # Make an x tick every 20 epochs
            ax.set_xticks(range(0, len(curve), 50))
            # Make the x-ticks labels be the number of the epoch
            ax.set_xticklabels(range(0, len(curve)* 100, 5000) )
            ax.set_yscale('log')
            ax.set_title(labels[i])

        # plot the lr_df in axes[1,0]
        lr_df.plot(ax=axes[1,0])
        axes[1,0].set_title("Learning rate")
        axes[1,0].set_xticks(range(0, len(lr_df), 50))
        axes[1,0].set_xticklabels(range(0, len(lr_df)* 100, 5000) )
        axes[1,0].set_yscale('log')
        axes[1,0].legend()

        # plot the Q10_log in axes[1,1]
        Q10_log.T.plot(ax=axes[1,1])
        axes[1,1].set_title("Q10")
        axes[1,1].set_xticks(range(0, len(Q10_log.T), 50))
        axes[1,1].set_xticklabels(range(0, len(Q10_log.T)* 100, 5000) )
        axes[1,1].get_legend().remove()

        # Make an x tick every 20 epochs
        ax.set_xticks(range(0, len(curve), 50))
        # Make the x-ticks labels be the number of the epoch
        ax.set_xticklabels(range(0, len(curve)* 100, 5000) )
        ax.set_yscale('log')
        ax.set_title(labels[i])


        fig.suptitle("Training curves")
        if results_path:
            fig.savefig(results_path.joinpath('training_curves_set_wise.pdf'))

def monthly_curves(flux, data, compare_to=['orth', 'ann', 'DT', 'NT'], results_path=None):
    
    # LOAD DATA from 
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    colors = ['#d73027', '#fc8d59','#91bfdb','tab:green']
    line_styles = ['-', '--', '-.', '-']

    fig, axes = plt.subplots(12,1, figsize=(40,50), sharex = True, sharey=True)
    
    for month, ax in enumerate(axes.flatten()):
        df_temp = data[data["Month"] == month+1]
        mean = df_temp[flux + "_mean"]
        std = df_temp[flux +"_std"]
        df_temp = df_temp.sort_values("tom")
        # Check if there are duplicates in df_temp['tom']
        if df_temp['tom'].duplicated().any():
            print("Duplicates in tom")
        
        ax.fill_between(df_temp["tom"].unique(), mean.values - std*1.96, mean.values + std*1.96, color = 'blue', alpha=0.2)
        ax.plot(df_temp["tom"].unique(), mean.values, color = 'blue', label = "Mean")
        ax.set_title(month_names[month])
        
        # Make xticks time of month tom
        ax.set_xticks(df_temp["tom"].unique()[::48])
        # Make every 48th tick a label that corresponds to the day
        ax.set_xticklabels(df_temp["dom"][::48], rotation=90)
        
        
        for i, compare in enumerate(compare_to):
            ax.plot(df_temp[f'{flux}_{compare}'].values, color = colors[i], linestyle=line_styles[i] , label = compare)
        
        for i, qc in enumerate(df_temp['NEE_QC']):
            if qc == 0:
                ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.3, lw=0)
        
        #Todo: make limits depending on max and min values
        if flux == 'NEE':
            ax.set_ylim([-25, 20])
        elif flux == 'GPP': 
            ax.set_ylim([-5, 30])
        elif flux == 'RECO':
            ax.set_ylim([-5, 30])
        
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.45, 0.90), fontsize=24)

    fig.suptitle(f'Comparison of the predicted {flux} in monthly curves for different flux partitioning methods.', fontsize=24)
    #fig.tight_layout()
    if results_path:
        fig.savefig(f'{results_path}/{flux}.pdf', bbox_inches='tight', facecolor='white',  transparent=False)   

import numpy as np
import matplotlib.pyplot as plt
from skill_metrics import taylor_diagram

def taylor_plot(NEE, GPP, RECO, ensemble_size, filtered=True, results_path=None):
#filtered = True
#ensemble_size = 100
#obs_label = 'NEE_NT'
    data = {'NEE': NEE, 'GPP': GPP, 'RECO': RECO}
            
    compare_to = ['mean', 'orth', 'ann', 'DT', 'NT']
    fig, axes = plt.subplots(2,3, figsize=(18,12))
    
    for j, (obs, flux) in enumerate(product(['NT', 'DT'], ['NEE', 'GPP', 'RECO'])):
        compare_to.remove(obs)
        flux_data = data[flux]
        ax = axes.flatten()[j]
        if filtered:
            mask = flux_data['NEE_QC'] == 0
        else:
            True

        # Generate some dummy data
        observed = flux_data[flux+'_'+obs].values[mask]
        models = [flux_data[str(i)].values[mask] for i in range(ensemble_size)] + [flux_data[method].values[mask] for method in [flux+'_'+method for method in compare_to]]

        # Compute standard deviations and correlation coefficients
        stddevs = np.array([np.std(observed), *[np.std(model) for model in models]])

        # Compute centered root mean squared error for each model to the observed data
        observations_centered = observed - np.mean(observed)
        models_centered = [model - np.mean(model) for model in models]
        RMSE_centered = [mean_squared_error(observations_centered, model_centered, squared=False) for model_centered in models_centered]
        CRMSES = np.array([0] + RMSE_centered)

        # Compute correlation coefficient
        corrcoefs = np.array([1]+[*[np.corrcoef(observed, model)[0, 1] for model in models]])

        # Create the Taylor diagram

        labels = ['mean', 'DT', 'ann', 'orth']
        colors = ['black', 'black', 'black','black']
        markers = ['x', 'x', 'x', 'x']

        # Add title to subplot
        ax.set_title(f'{flux} {obs}', fontsize=16)

        taylor_diagram(ax, stddevs[:ensemble_size+1], CRMSES[:ensemble_size+1], corrcoefs[:ensemble_size+1], markerLabel=['Observation', *([f'' for i in range(ensemble_size)])], styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation', 
                    markerColor='orange', alpha=0.5, markersymbol='.', markerSize=3, colRMS='red', widthRMS=2.0, 
                    checkstats='on')
        for i in range(4):
            taylor_diagram(ax, stddevs[[0,ensemble_size+1+i]], CRMSES[[0,ensemble_size+1+i]], corrcoefs[[0,ensemble_size+1+i]], markerLabel=['Observation',labels[i]], styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation', 
                        markerColor=colors[i], markersymbol=markers[i], markerSize=6, colRMS='red', widthRMS=2.0, 
                        checkstats='on')
        compare_to = compare_to + [obs]
        
    fig.suptitle(f'Taylor plot of estimations over ensemble and other models.', fontsize=12)

    if results_path:
        fig.savefig(f'{results_path}/Taylor_diagram.pdf', bbox_inches='tight', facecolor='white',  transparent=False)   
