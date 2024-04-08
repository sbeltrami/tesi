import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#funzione che crea un vettore di pesi per poli/equatore e ritorna un dataset pesato
#in input ci deve essere il dataset con []
def compute_dataset_weighted(dataset):
    weights = np.cos(np.deg2rad(dataset.lat))
    dataset_weighted = dataset.weighted(weights)
    return dataset_weighted

#funzione per il calcolo della media annuale pesata, a partire da un dataset con dati ogni mese
def compute_annual_mean_weighted(dataset_weighted):
    annual_mean_weighted = dataset_weighted.mean(dim=("lon", "lat")).resample(time='YE').mean(dim='time')
    return annual_mean_weighted

#trasformazione in °C di un dataset in °K. Il dataset in input con []
def convert_dataset_celsius(dataset):
    dataset_celsius = dataset - 273.15
    return dataset_celsius

#funzione per il calcolo delle anomalie --> attenzione bisogna usarla dopo aver creato una climatologia su 30 anni
def compute_anomaly(annual_mean_weighted, annual_mean_weighted_30):
    climatological_mean = annual_mean_weighted_30.mean()
    anomaly = annual_mean_weighted - climatological_mean
    return anomaly 

#funzione che prende un dataset (in °C e già con []), un tempo di inizio e di fine per fare il resample, un indice i e calcola la media di tutti i mesi dei raggruppamenti, partendo dal trimestre di marzo
def compute_mean_resample_mar(dataset,year_start,year_end,i):
    dataset_resample = dataset.sel(time=slice(year_start,year_end)).resample(time='QE-MAR').mean(dim='time')
    resample_mean = dataset_resample.sel(time=dataset_resample['time.month']==((i+1)*3)).mean(dim='time')
    return resample_mean

#Posso cancellare?

#funzione che calcola il periodo Dicembre, Gennaio, Febbraio dello stesso inverno
#dataset già convertito in °C e già con []
def create_djf_dataset(dataset,year_december):   
    djf_data = dataset.sel(time=((dataset['time.month'] == 12) & (dataset['time.year'] >= year_december)
    ) | (((dataset['time.month'] >= 1) & (dataset['time.month'] <= 2)) 
    & (dataset['time.year'] >= (year_december + 1))))
    return djf_data

#funzione che calcola la media temporale per ogni stagione MAM, JJA, SON
#dataset è già con il periodo selezionato e, se non modello, già in °C
def compute_mean_time_season(dataset,seas):
    time_mean_seas = dataset.groupby('time.season').mean(dim='time').sel(season=seas)
    return time_mean_seas

#TOS
#funzione per il plot dei bias tos
def plot_bias_tos(n_rows,n_cols,fig_size,number_models,name_models_to_plot,name_dict,val_min,val_max): #number_models è la lista che riporta il numero di modelli in un determinato cluster
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size)
    # Plot dei modelli
    for i in range(n_rows): #ciclo sulle righe 
        for j in range(n_cols): #ciclo sulle colonne 
            models_index_list = i*n_cols + j #indice del modello all'interno della lista
            if models_index_list == number_models:
                break
            model_name = name_models_to_plot[models_index_list]
            plot_mod = name_dict[model_name]['North Atlantic bias DJF'].plot.pcolormesh(ax=ax[i, j])
            # Fisso la scala
            plot_mod.set_clim(vmin=val_min, vmax=val_max)
            #ax[i,j].legend(loc='upper right')
            ax[i,j].set_ylabel('latitude')
            ax[i,j].set_xlabel('longitude')
            
            # Aggiungi il nome del modello in alto a destra
            ax[i,j].text(0.95, 0.95, model_name, horizontalalignment='right', verticalalignment='top', transform=ax[i,j].transAxes, fontsize=10, color='black')

    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= number_models:
                ax[i, j].axis('off')
    #fig.tight_layout()

#plot dei bias dei 2 modelli
def plot_bias_2_models_tos(fig_size,name_models_to_plot,name_dict):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 righe
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        plot_mod = name_dict[model_name]['North Atlantic bias DJF'].plot.pcolormesh(ax=ax[i])
        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')
            
        # Aggiungi il nome del modello in alto a destra
        ax[i].text(0.95, 0.95, model_name, horizontalalignment='right', verticalalignment='top', transform=ax[i].transAxes, fontsize=10, color='black')


    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')
    
    #fig.tight_layout()



#funzione per il plot dei cluster medi tos
def plot_mean_cluster_tos(number_models,name_models_to_plot,name_dict):
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(number_models):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / number_models
    #plot del valor medio per il cluster 3
    plot_mod = mean_bias.plot.pcolormesh()
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.tight_layout()


#ATMOS
#funzione per plot bias atmos
def plot_bias_atmos(n_rows,n_cols,fig_size,name_models_to_plot,name_dict,dataset_seas_mean): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli, name_dict è model_atmos, val_min e max sono i valori che fissano la scala, dataset_seas_mean è era_na_seas_mean
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size)

    for i in range(n_rows): #ciclo sulle righe
        for j in range(n_cols): #ciclo sulle colonne
            models_index_list = i * n_cols + j #indice del modello all'interno della lista
            if models_index_list == len(name_models_to_plot):
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['atmos North Atlantic bias DJF']
            plot_mod = ax[i,j].pcolormesh(data_array[0], cmap='coolwarm')            # Fisso la scala
            #plot_mod.set_clim(vmin=val_min, vmax=val_max)

            #Plot della climatologia dei singoli mdoelli e di ERA5
            data = name_dict[model_name]['atmos North Atlantic bias DJF']     
            data_era = dataset_seas_mean[3]
            #plot
            ax[i,j].contour(data[0], levels=5, linewidths=0.5, linestyles='dashed', colors='k')
            ax[i,j].contour(data_era[0], levels=5, linewidths=0.5, linestyles='dashdot', colors='g')
            
            # Fisso la scala
            #plot_mod.set_clim(vmin=-6, vmax=6)
            ax[i,j].set_ylabel('lat')
            ax[i,j].set_xlabel('lon')
            
            # Aggiungi il nome del modello in alto a destra
            ax[i,j].text(0.95, 0.95, model_name, horizontalalignment='right', verticalalignment='top', transform=ax[i,j].transAxes, fontsize=10, color='black')

    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= len(name_models_to_plot):
                ax[i, j].axis('off')
    
    cbar = fig.colorbar(plot_mod, ax=ax.ravel().tolist(), orientation='vertical')
    # Per fissare i colori sulla scala di colori
    #cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6])

    # Legenda per linee tratteggiate
    fig.text(1, 1, ['Linee nere - modelli', 'Linee verdi - ERA5'], horizontalalignment='right', verticalalignment='top', fontsize=10, color='black')
    #fig.tight_layout()

#plot del bias per due modelli
def plot_bias_2_models_atmos(fig_size,name_models_to_plot,name_dict,dataset_seas_mean):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 righe
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        data_array = name_dict[model_name]['atmos North Atlantic bias DJF']
        plot_mod = ax[i].pcolormesh(data_array[0],cmap='coolwarm')

        #Plot della climatologia dei singoli mdoelli e di ERA5
        data = name_dict[model_name]['atmos North Atlantic seasonal mean']        
        data_era = dataset_seas_mean[3]
        #plot
        ax[i].contour(data[0], levels=5, linewidths=0.5, linestyles='dashed', colors='k')
        ax[i].contour(data_era[0], levels=5, linewidths=0.5, linestyles='dashdot', colors='g')

        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')        
        # Aggiungi il nome del modello in alto a destra
        ax[i].text(0.95, 0.95, model_name, horizontalalignment='right', verticalalignment='top', transform=ax[i].transAxes, fontsize=10, color='black')

    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')

    cbar = fig.colorbar(plot_mod, ax=ax.ravel().tolist(), orientation='vertical')
    # Per fissare i colori sulla scala di colori
    #cbar.set_ticks([-6, -4, -2, 0, 2, 4, 6])       
    # Legenda per linee tratteggiate
    fig.text(1, 1, ['Linee nere - modelli', 'Linee verdi - ERA5'], horizontalalignment='right', verticalalignment='top', fontsize=10, color='black')
    #fig.tight_layout()


#plot dei cluster medi atmos
def plot_mean_cluster_atmos(name_models_to_plot,name_dict): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli da plottare, name_dict è o models_atmos
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_models_to_plot)):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_models_to_plot)
    #plot del valor medio per il cluster 3
    plot_mod = mean_bias[0].plot.pcolormesh()
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.tight_layout()
