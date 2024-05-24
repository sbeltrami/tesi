import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import random
import math
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

#Funzione che calcola la media di un dataset xarray
def compute_mean(name_list,name_dict):
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias_cluster0 = sum_bias / len(name_list)
    return mean_bias_cluster0
def compute_mean_zonmean(name_list,name_dict):
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        sum_bias = sum_bias + name_dict[model_name]['zonmean bias DJF']
    #valor medio
    mean_bias_cluster0 = sum_bias / len(name_list)
    return mean_bias_cluster0

#TOS
#funzione per il plot dei bias tos
def plot_bias_tos(n_rows,n_cols,fig_size,number_models,name_models_to_plot,name_dict,val_min,val_max,title_plot,title_pdf): #number_models è la lista che riporta il numero di modelli in un determinato cluster
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    # Plot dei modelli
    for i in range(n_rows): #ciclo sulle righe 
        for j in range(n_cols): #ciclo sulle colonne 
            models_index_list = i*n_cols + j #indice del modello all'interno della lista
            if models_index_list == number_models:
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['North Atlantic bias DJF']
            plot_mod = ax[i,j].pcolormesh(data_array.lon,data_array.lat,data_array,cmap='seismic',vmin=val_min, vmax=val_max)
            #plot di cartopy
            ax[i,j].coastlines()
            ax[i,j].set_extent([data_array.lon.min(), data_array.lon.max(), data_array.lat.min(), data_array.lat.max()], crs=ccrs.PlateCarree())
            #label assi
            ax[i,j].set_ylabel('latitude')
            ax[i,j].set_xlabel('longitude')
            ax[i,j].set_title(model_name) #nome di ogni singolo modello sul plot
            #ax[i, j].set_xticks(np.arange(data_array.lon.min(), data_array.lon.max(), 20))
            #ax[i, j].set_yticks(np.arange(data_array.lat.min(), data_array.lat.max(), 10))


    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= number_models:
                ax[i, j].axis('off')
    
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#plot dei bias dei 2 modelli
def plot_bias_2_models_tos(fig_size,val_min,val_max,name_models_to_plot,name_dict,title_plot,title_pdf):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 righe
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        data_array = name_dict[model_name]['North Atlantic bias DJF']
        plot_mod = ax[i].pcolormesh(data_array.lon,data_array.lat,data_array,cmap='seismic',vmin=val_min, vmax=val_max)
        #plot di cartopy
        ax[i].coastlines() 
        ax[i].set_extent([data_array.lon.min(), data_array.lon.max(), data_array.lat.min(), data_array.lat.max()], crs=ccrs.PlateCarree())
        #valori assi            
        #ax[i].set_xticks(np.arange(data_array.lon.min(),data_array.lon.max(), 20), crs=ccrs.PlateCarree())
        #ax[i].set_yticks(np.arange(data_array.lat.min(),data_array.lat.max(), 10), crs=ccrs.PlateCarree())
        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')
        ax[i].set_title(model_name) #nome di ogni singolo modello sul plot


    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')
    
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40) 
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#funzione per il plot dei cluster medi tos

def plot_mean_cluster_tos(number_models,name_models_to_plot,name_dict,title_plot,title_pdf,v_min,v_max,fig_size):
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(number_models):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / number_models
    #plot del valor medio per il cluster
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})
    mean_bias = mean_bias.assign_coords(lon=(np.where(mean_bias.lon >= 280, mean_bias.lon - 360, mean_bias.lon))) #metto i valori negativi di lon per mean_bias
    plot_mod = ax.pcolormesh(mean_bias.lon,mean_bias.lat,mean_bias,vmin=v_min, vmax=v_max,cmap='seismic')  #trasformazione cartografica = lonxlat
    ax.coastlines() #gca = get current axis
    #valori assi            
    ax.set_xticks(np.arange(mean_bias.lon.min(),mean_bias.lon.max(), 20))
    ax.set_yticks(np.arange(mean_bias.lat.min(),mean_bias.lat.max(), 10))
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')
    # Titolo
    fig.colorbar(plot_mod)
    fig.suptitle(title_plot, fontsize=16, y=1.02)

    fig.savefig(title_pdf, format='pdf')

#plot della standard deviation dei cluster di tos
def plot_std_cluster_tos(name_models_to_plot,name_dict,v_min,v_max,title_plot,title_pdf): #name_dict è models_tos
    dataset = [] #Inizializzo una lista
    for i in range(len(name_models_to_plot)): #Vado ad inserire all'interno di dataset tutti gli elementi 'North Atlantic bias DJF' della j-esima lista, dove j = 0,...,4
        dataset.append(name_dict[name_models_to_plot[i]]['North Atlantic bias DJF'])

    combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
    std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
    # Plot
    
    std_dev = std_dev.assign_coords(lon=(np.where(std_dev.lon >= 280, std_dev.lon - 360, std_dev.lon))) #metto i valori negativi di lon per mean_bias
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max,
    subplot_kws={"projection":ccrs.PlateCarree()})  #trasformazione cartografica = lonxlat    
    #valori assi            
    plt.xticks(np.arange(std_dev.lon.min(),std_dev.lon.max(), 20))
    plt.yticks(np.arange(std_dev.lat.min(),std_dev.lat.max(), 10))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.gca().coastlines() #gca = get current axis
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

def plot_5_mean_cluster_tos(list_5_clusters,fig_size,name_dict,v_min,v_max,title_plot,title_pdf): #funzione per il plot dei 5 cluster medi di tos
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=fig_size,subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    #calcolo il valor medio
    for j in range(len(list_5_clusters)): #ciclo su tutti i 5 cluster medi
        #Inizializzo sum_bias per il calcolo della media di tos
        sum_bias = 0
        for i in range(len(list_5_clusters[j])): #ciclo sui modelli del cluster j-esimo
            model_name = list_5_clusters[j][i]
            sum_bias = sum_bias + name_dict[model_name]['North Atlantic bias DJF']
        #valor medio
        mean_bias_tos = sum_bias / len(list_5_clusters[j])
        mean_bias_tos = mean_bias_tos.assign_coords(lon=(np.where(mean_bias_tos.lon >= 280, mean_bias_tos.lon - 360, mean_bias_tos.lon))) #metto i valori negativi di lon per mean_bias
        #plot
        if j <= 2: #primi 3 cluster medi
            k = 0 #indice per le righe --> prima riga
            l = j #indice per le colonne
        else:
            k = 1 #indice per le righe --> seconda riga
            l = k*j - 3  #indice per le colonne --> l appartiene [0,1]
        plot_mod = ax[k,l].pcolormesh(mean_bias_tos.lon,mean_bias_tos.lat,mean_bias_tos,vmin=v_min, vmax=v_max,cmap='seismic')  #trasformazione cartografica = lonxlat
        ax[k,l].coastlines() #gca = get current axis
        #valori assi            
        ax[k,l].set_xticks(np.arange(mean_bias_tos.lon.min(),mean_bias_tos.lon.max(), 20))
        ax[k,l].set_yticks(np.arange(mean_bias_tos.lat.min(),mean_bias_tos.lat.max(), 10))
        ax[k,l].set_ylabel('latitude')
        ax[k,l].set_xlabel('longitude')
        ax[k,l].set_title(f'Cluster {j}', fontsize=16, y=1.02)

    #rimuovo il plot vuoto ax[2,2]
    ax[1, 2].axis('off')
    # Titolo
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

def plot_5_std_cluster_tos(list_5_clusters,name_dict,fig_size,v_min,v_max,title_plot,title_pdf): #funzione che plotta la std dei 5 cluster di tos
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=fig_size,subplot_kw={"projection": ccrs.PlateCarree()}) #trasformazione cartografica = lonxlat   
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    #calcolo il valor medio
    for j in range(len(list_5_clusters)-1): #ciclo su tutti i 5 cluster medi
        dataset = []
        for i in range(len(list_5_clusters[j])): #ciclo sui modelli del cluster j-esimo
            dataset.append(name_dict[list_5_clusters[j][i]]['North Atlantic bias DJF'])
        combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
        std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
        std_dev = std_dev.assign_coords(lon=(np.where(std_dev.lon >= 280, std_dev.lon - 360, std_dev.lon))) #metto i valori negativi di lon per mean_bias
        #plot
        if j <= 2: #primi 3 cluster medi
            k = 0 #indice per le righe --> prima riga
            l = j #indice per le colonne
        else:
            k = 1 #indice per le righe --> seconda riga
            l = k*j - 3  #indice per le colonne --> l appartiene [0,1]
        plot_mod = std_dev.plot(ax=ax[k,l],cmap='Reds',vmin=v_min,vmax=v_max, add_colorbar=False) 
        #valori assi            
        ax[k,l].set_xticks(np.arange(std_dev.lon.min(),std_dev.lon.max(), 20))
        ax[k,l].set_yticks(np.arange(std_dev.lat.min(),std_dev.lat.max(), 10))
        ax[k,l].set_ylabel('latitude')
        ax[k,l].set_xlabel('longitude')
        ax[k,l].coastlines() #gca = get current axis        
        ax[k,l].set_title(f'Cluster {j}', fontsize=16, y=1.02)

    ax[1,1].axis('off')
    ax[1,2].axis('off')
    # Titolo
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#ATMOS
#funzione per plot bias atmos
#funzione per plot bias atmos
def plot_bias_atmos(n_rows,n_cols,fig_size,v_min,v_max,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli, name_dict è model_atmos, val_min e max sono i valori che fissano la scala, dataset_seas_mean è era_na_seas_mean
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size,subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots

    for i in range(n_rows): #ciclo sulle righe
        for j in range(n_cols): #ciclo sulle colonne
            models_index_list = i * n_cols + j #indice del modello all'interno della lista
            if models_index_list == len(name_models_to_plot):
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['atmos North Atlantic bias DJF']    
            plot_mod = ax[i,j].pcolormesh(data_array[0].lon, data_array[0].lat, data_array[0], cmap='seismic',  vmin=v_min, vmax=v_max)
            #Plot della climatologia dei singoli mdoelli e di ERA5
            data = name_dict[model_name]['atmos North Atlantic seasonal mean DJF']     
            data_era = dataset_seas_mean[4]
            #plot
            contour = data[0].plot.contour(ax=ax[i,j],colors='k')
            ax[i,j].clabel(contour, fmt='%1.1f')
            #data_era[0].plot.contour(ax=ax[i,j],colors='g')
            contour_era = data_era[0].plot.contour(ax=ax[i,j], colors='g')
            ax[i,j].clabel(contour_era, fmt='%1.1f')
            #plot di cartopy
            ax[i,j].coastlines()  
            #valori assi            
            ax[i, j].set_xticks(np.arange(data_array[0].lon.min(),data_array[0].lon.max(),20))
            ax[i, j].set_yticks(np.arange(data_array[0].lat.min(),data_array[0].lat.max(),10))
            #label assi
            ax[i,j].set_ylabel('lat')
            ax[i,j].set_xlabel('lon')
            ax[i,j].set_title(model_name) #nome di ogni singolo modello sul plot

    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= len(name_models_to_plot):
                ax[i, j].axis('off')
    
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#plot del bias per due modelli
def plot_bias_2_models_atmos(fig_size,v_min,v_max,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 colonne
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        data_array = name_dict[model_name]['atmos North Atlantic bias DJF']
        plot_mod = ax[i].pcolormesh(data_array[0].lon, data_array[0].lat, data_array[0], cmap='seismic', vmin=v_min, vmax=v_max)
        #Plot della climatologia dei singoli mdoelli e di ERA5
        data = name_dict[model_name]['atmos North Atlantic seasonal mean DJF']        
        data_era = dataset_seas_mean[4]
        #plot
        contour_data = data[0].plot.contour(ax=ax[i],colors='k')
        ax[i].clabel(contour_data, fmt='%1.1f')
        contour_era = data_era[0].plot.contour(ax=ax[i],colors='g')
        ax[i].clabel(contour_era, fmt='%1.1f')
        #plot di cartopy
        ax[i].coastlines() 
        #valori assi            
        ax[i].set_xticks(np.arange(data_array[0].lon.min(),data_array[0].lon.max(), 20))
        ax[i].set_yticks(np.arange(data_array[0].lat.min(),data_array[0].lat.max(), 10))
        #label assi
        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')
        ax[i].set_title(model_name) #nome di ogni singolo modello sul plot

    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off') 

    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#plot dei cluster medi atmos
def plot_mean_cluster_atmos(name_models_to_plot, name_dict, dataset_seas_mean, title_plot,title_pdf, v_min, v_max, fig_size):
    sum_bias = 0
    for i in range(len(name_models_to_plot)):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    mean_bias = sum_bias / len(name_models_to_plot)
    
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_mod = ax.pcolormesh(mean_bias[0].lon, mean_bias[0].lat, mean_bias[0], vmin=v_min, vmax=v_max, cmap='seismic')
    data_era = dataset_seas_mean[4]
    contour_era = ax.contour(data_era[0].lon, data_era[0].lat, data_era[0], colors='g')
    ax.clabel(contour_era, fmt='%1.1f')
    ax.coastlines()
    #valori assi            
    ax.set_xticks(np.arange(mean_bias[0].lon.min(),mean_bias[0].lon.max(), 20))
    ax.set_yticks(np.arange(mean_bias[0].lat.min(),mean_bias[0].lat.max(), 10))
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')
    # Legenda per linee verdi
    fig.legend(['Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    fig.colorbar(plot_mod)
    fig.suptitle(title_plot, fontsize=16, y=1.02)

    fig.savefig(title_pdf, format='pdf')

#plot della standard deviation dei cluster di atmos
def plot_std_cluster_atmos(name_models_to_plot,name_dict,v_min,v_max,title_plot,title_pdf): #name_dict è models_atmos
    dataset = [] #Inizializzo una lista
    for i in range(len(name_models_to_plot)): #Vado ad inserire all'interno di dataset tutti gli elementi 'atmos North Atlantic bias DJF' della j-esima lista, dove j = 0,...,4
        dataset.append(name_dict[name_models_to_plot[i]]['atmos North Atlantic bias DJF'])

    combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
    std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
    # Plot
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max,
    subplot_kws={"projection":ccrs.PlateCarree()})  #trasformazione cartografica = lonxlat        
    #valori assi            
    plt.xticks(np.arange(std_dev[0].lon.min(),std_dev[0].lon.max(), 20))
    plt.yticks(np.arange(std_dev[0].lat.min(),std_dev[0].lat.max(), 10))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.gca().coastlines() #gca = get current axis)
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

def plot_5_mean_cluster_atmos(list_5_clusters, dataset_seas_mean, fig_size, name_dict, v_min, v_max, title_plot, title_pdf):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots    
    # Inizializzo sum_bias per il calcolo della media di atmos
    for j in range(len(list_5_clusters)):  # ciclo su tutti i 5 cluster medi
        sum_bias = 0
        for i in range(len(list_5_clusters[j])):  # ciclo sui modelli del cluster j-esimo
            model_name = list_5_clusters[j][i]
            sum_bias += name_dict[model_name]['atmos North Atlantic bias DJF']        
        # Calcolo del valor medio
        mean_bias_atmos = sum_bias / len(list_5_clusters[j])            
        # Plot
        if j <= 2:  # primi 3 cluster medi
            k = 0  # indice per le righe --> prima riga
            l = j  # indice per le colonne
        else:
            k = 1  # indice per le righe --> seconda riga
            l = k*j - 3  # indice per le colonne --> l appartiene [0, 1]        
        plot_mod = ax[k, l].pcolormesh(mean_bias_atmos[0].lon, mean_bias_atmos[0].lat, mean_bias_atmos[0], vmin=v_min, vmax=v_max, cmap='seismic')
        data_era = dataset_seas_mean[4]
        contour_era = ax[k, l].contour(data_era[0].lon, data_era[0].lat, data_era[0], colors='g')
        ax[k,l].clabel(contour_era, fmt='%1.1f')
        ax[k, l].coastlines()        
        # Valori assi            
        ax[k, l].set_xticks(np.arange(mean_bias_atmos[0].lon.min(), mean_bias_atmos[0].lon.max(), 20))
        ax[k, l].set_yticks(np.arange(mean_bias_atmos[0].lat.min(), mean_bias_atmos[0].lat.max(), 10))
        ax[k, l].set_xlabel('longitude')
        ax[k, l].set_ylabel('latitude')
        ax[k, l].set_title(f'Cluster {j}', fontsize=16, y=1.02)
    # Rimuovo il plot vuoto ax[1, 2]
    ax[1, 2].axis('off')    
    # Legenda per linee verdi
    fig.legend(['Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))    
    # Barra del colore
    fig.colorbar(plot_mod, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.6, aspect=40)    
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

def plot_5_std_cluster_atmos(list_5_clusters,name_dict,fig_size,v_min,v_max,title_plot,title_pdf): #funzione che plotta la std dei 5 cluster di tos
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=fig_size,subplot_kw={"projection": ccrs.PlateCarree()}) #trasformazione cartografica = lonxlat   
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    #calcolo il valor medio
    for j in range(len(list_5_clusters)-1): #ciclo su tutti i 5 cluster medi
        dataset = []
        for i in range(len(list_5_clusters[j])): #ciclo sui modelli del cluster j-esimo
            dataset.append(name_dict[list_5_clusters[j][i]]['atmos North Atlantic bias DJF'])
        combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
        std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
        #plot
        if j <= 2: #primi 3 cluster medi
            k = 0 #indice per le righe --> prima riga
            l = j #indice per le colonne
        else:
            k = 1 #indice per le righe --> seconda riga
            l = k*j - 3  #indice per le colonne --> l appartiene [0,1]
        plot_mod = std_dev.plot(ax=ax[k,l],cmap='Reds',vmin=v_min,vmax=v_max, add_colorbar=False) 
        #valori assi          
        ax[k,l].set_xticks(np.arange(std_dev[0].lon.min(),std_dev[0].lon.max(), 20))
        ax[k,l].set_yticks(np.arange(std_dev[0].lat.min(),std_dev[0].lat.max(), 10))  
        ax[k,l].set_ylabel('latitude')
        ax[k,l].set_xlabel('longitude')
        ax[k,l].coastlines() #gca = get current axis        
        ax[k,l].set_title(f'Cluster {j}', fontsize=16, y=1.02)

    ax[1,1].axis('off')
    ax[1,2].axis('off')
    # Barra del colore
    fig.colorbar(plot_mod, ax=ax.ravel().tolist(), orientation='horizontal', shrink=0.6, aspect=40)
    # Titolo 
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#ZONMEAN
#plot medie zonali
#plot medie zonali
def plot_zonmean(n_rows, n_cols, fig_size, name_models_to_plot, name_dict, dataset_seas_mean, v_min, v_max, title_plot, title_pdf):
    # Plot medie annuali dei modelli
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    # Plot dei modelli
    for i in range(n_rows):  # Ciclo sulle righe
        for j in range(n_cols):  # Ciclo sulle colonne
            models_index_list = i * n_cols + j  # Indice del modello all'interno della lista
            if models_index_list == len(name_models_to_plot):
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['zonmean bias DJF']
            plot_mod = ax[i, j].pcolormesh(data_array.lat, data_array.plev, data_array.sel(lon=0), cmap='seismic', vmin=v_min, vmax=v_max)
            # Plot della climatologia dei singoli modelli e di ERA5
            data = name_dict[model_name]['zonmean seasonal mean DJF']
            data_era = dataset_seas_mean[4]
            # Plot
            contour_data = data.sel(lon=0).plot.contour(ax=ax[i, j], colors='k')  # In modo che l'array sia 2D su plev e lat
            ax[i,j].clabel(contour_data, fmt='%1.1f')
            contour_era = data_era.sel(lon=0).plot.contour(ax=ax[i, j], colors='g')
            ax[i,j].clabel(contour_era, fmt='%1.1f')
            ax[i, j].set_ylabel('plev')
            ax[i, j].set_xlabel('lat')
            ax[i, j].set_title(model_name)  # Nome di ogni singolo modello sul plot
            # Inverto l'asse y in modo che i livelli di pressione siano corretti
            ax[i, j].invert_yaxis()

    # Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j            
            if models_index_list >= len(name_models_to_plot):
                ax[i, j].axis('off')

    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    fig.savefig(title_pdf, format='pdf')


#plot medie zonali per due cluster
def plot_zonmean_2_cluster(fig_size,name_models_to_plot,name_dict,dataset_seas_mean,v_min,v_max,title_plot,title_pdf):
    #plot medie annuali dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    # Plot dei modelli
    for i in range(2): #ciclo sulle colonne
        model_name = name_models_to_plot[i]
        data_array = name_dict[model_name]['zonmean bias DJF']
        plot_mod = ax[i].pcolormesh(data_array.lat, data_array.plev, data_array.sel(lon=0), cmap='seismic', vmin=v_min, vmax=v_max)     
        #Plot della climatologia dei singoli mdoelli e di ERA5
        data = name_dict[model_name]['zonmean seasonal mean DJF']        
        data_era = dataset_seas_mean[4]
        #plot
        contour_data = data.sel(lon=0).plot.contour(ax=ax[i],colors='k')
        ax[i].clabel(contour_data, fmt='%1.1f')
        contour_era = data_era.sel(lon=0).plot.contour(ax=ax[i],colors='g')
        ax[i].clabel(contour_era, fmt='%1.1f')
        ax[i].set_ylabel('plev')
        ax[i].set_xlabel('lat')
        ax[i].set_title(model_name) #nome di ogni singolo modello sul plot
        # Inverto l'asse y in modo t.c i livelli di pressione siano corretti
        ax[i].invert_yaxis()

    # Rimuovo i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')  
    
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')   

#plot dei cluster medi per medie zonali

#funzione per il plot dei cluster medi tos
def plot_mean_cluster_zonmean(number_models,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf,v_min,v_max,fig_size):
    #Inizializzo sum_zonmean per il calcolo della media
    sum_zonmean = 0
    #calcolo il valor medio
    for i in range(number_models):
        model_name = name_models_to_plot[i]
        zonmean = name_dict[model_name]['zonmean bias DJF']
        zonmean = zonmean.assign_coords({"plev" : zonmean.plev.round()}) #arrotondo in modo tale che i livelli di pressione siano gli stessi per ogni modello
        sum_zonmean = sum_zonmean + zonmean
    #valor medio
    mean_zonmean = sum_zonmean / number_models
    #plot del valor medio
    fig,ax = plt.subplots(figsize=fig_size)
    mean_zonmean.plot(vmin=v_min, vmax=v_max,cmap='seismic', ax=ax) 
    data_era = dataset_seas_mean[4]
    # Plot
    contour_era = data_era.sel(lon=0).plot.contour(colors='g',ax=ax)
    ax.clabel(contour_era, fmt='%1.1f')
    ax.set_ylabel('plev')
    ax.set_xlabel('latitude')
    ax.invert_yaxis()
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    # Legenda per linee verdi
    fig.legend(['Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    fig.savefig(title_pdf, format='pdf')

#plot della standard deviation dei cluster di zonmean
def plot_std_cluster_zonmean(name_models_to_plot,name_dict,v_min,v_max,title_plot,title_pdf): #name_dict è models_zonmean
    dataset = [] #Inizializzo una lista
    for i in range(len(name_models_to_plot)): #Vado ad inserire all'interno di dataset tutti gli elementi 'zonmean bias DJF' della j-esima lista, dove j = 0,...,4
        dataset.append(name_dict[name_models_to_plot[i]]['zonmean bias DJF'])

    combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
    std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
    # Plot
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max)
    plt.ylabel('plev')
    plt.xlabel('latitude')
    plt.gca().invert_yaxis()
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

def plot_5_mean_cluster_zonmean(list_5_clusters, dataset_seas_mean, fig_size, name_dict, v_min, v_max, title_plot, title_pdf): #plot dei 5 cluster medi di zonmean    
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=fig_size)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    #calcolo il valor medio
    for j in range(len(list_5_clusters)): # ciclo su tutti i 5 cluster medi
        #Inizializzo sum_zonmean per il calcolo della media di zonmean
        sum_zonmean = 0
        for i in range(len(list_5_clusters[j])):  # ciclo sui modelli del cluster j-esimo
            model_name = list_5_clusters[j][i]
            zonmean = name_dict[model_name]['zonmean bias DJF']
            zonmean = zonmean.assign_coords({"plev" : zonmean.plev.round()}) #arrotondo in modo tale che i livelli di pressione siano gli stessi per ogni modello
            sum_zonmean = sum_zonmean + zonmean
        #valor medio
        mean_zonmean = sum_zonmean / len(list_5_clusters[j])
        # Plot
        if j <= 2:  # primi 3 cluster medi
            k = 0  # indice per le righe --> prima riga
            l = j  # indice per le colonne
        else:
            k = 1  # indice per le righe --> seconda riga
            l = k*j - 3  # indice per le colonne --> l appartiene [0, 1]         
        plot_mod = mean_zonmean.plot(vmin=v_min, vmax=v_max, cmap='seismic', ax=ax[k,l], add_colorbar=False)
        data_era = dataset_seas_mean[4]
        contour_era = data_era.sel(lon=0).plot.contour(ax=ax[k,l],colors='g')
        ax[k,l].clabel(contour_era, fmt='%1.1f')
        ax[k,l].set_xlabel('latitude')
        ax[k,l].set_ylabel('plev')
        ax[k,l].invert_yaxis()
        # Titolo
        ax[k,l].set_title(f'Cluster {j}', fontsize=16, y=1.02)

    # Rimuovo il plot vuoto ax[1, 2]
    ax[1, 2].axis('off')    
    # Legenda per linee verdi
    fig.legend(['Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Barra del colore
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)          
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

def plot_5_std_cluster_zonmean(list_5_clusters,name_dict,fig_size,v_min,v_max,title_plot,title_pdf): #funzione che plotta la std dei 5 cluster di zonmean
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=fig_size) #trasformazione cartografica = lonxlat   
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    #calcolo il valor medio
    for j in range(len(list_5_clusters)-1): #ciclo su tutti i 5 cluster medi
        dataset = []
        for i in range(len(list_5_clusters[j])): #ciclo sui modelli del cluster j-esimo
            dataset.append(name_dict[list_5_clusters[j][i]]['zonmean bias DJF'])
        combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
        std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
        #plot
        if j <= 2: #primi 3 cluster medi
            k = 0 #indice per le righe --> prima riga
            l = j #indice per le colonne
        else:
            k = 1 #indice per le righe --> seconda riga
            l = k*j - 3  #indice per le colonne --> l appartiene [0,1]
        plot_mod = std_dev.plot(ax=ax[k,l],cmap='Reds',vmin=v_min,vmax=v_max, add_colorbar=False)  
        ax[k,l].set_ylabel('plev')
        ax[k,l].set_xlabel('latitude')
        ax[k,l].invert_yaxis()      
        ax[k,l].set_title(f'Cluster {j}', fontsize=16, y=1.02)

    ax[1,1].axis('off')
    ax[1,2].axis('off')
    # Barra del colore
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)   
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')
    
#BOOTRSTRAP 
#ATMOS
#funzione che estrae in modo random un numero di modelli e ne calcola il valor medio di bias DJF --> per un singolo cluster
def bs_sample_mean(n_iterations,name_dict,name_list): 
    #creazione di una lista con il nome dei modelli
    list_name_models_atmos = list(name_dict.keys())
    sample_rand = [] #inizializzazione lista di modelli presi in modo random, in numero n_iterations
    sample_mean = [] #inizializzazione di una lista in cui vado ad inserire la media di ogni sample, preso con l'estrazione random
    for n in range(n_iterations): #itero n_iterations volte
        sample_rand = random.sample(range(len(name_dict)), len(name_list)) #lista = estraggo 4 numeri random da 0 a 36, che sono il numero associato ad ogni modello
        sample_sum = 0 #inizializzazione ad ogni iterazione di sample_sum
        for i in range(len(name_list)): #ciclo sui 4 modelli presi
            sample_sum += name_dict[list_name_models_atmos[sample_rand[i]]]['atmos North Atlantic bias DJF']
        sample_mean.append(sample_sum/len(sample_rand)) #calcolo il valore medio per ogni cluster
    return sample_mean

#funzione che calcola l'array mean-std-2.5th-97.5th della distribuzione bootstrap per ogni grid cell
def bs_compute_array_mean_std_95cl(n_iterations,sample_mean): #95% confidence level
    #Inizializzazione array
    cell_grid_iteration = np.zeros(n_iterations) #array in cui metto i valori medi di un solo punto griglia per iterazioni diverse
    array_mean = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori medi delle distribuzioni per ogni punto griglia
    array_std = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude le deviazioni standard delle distribuzioni per ogni punto griglia
    array_2th_percentile = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori in cui si ha il 2.5th percentile delle distribuzioni per ogni pt griglia
    array_97th_percentile = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori in cui si ha il 97.5th percentile delle distribuzioni per ogni pt griglia
    #Determino mean, std, 2.5th-97.5th percentile delle distribuzioni per ogni pt griglia
    for i in range(len(sample_mean[0].lat)): #ciclo sulle latitudini
        for j in range(len(sample_mean[0].lon)): #ciclo sulle longitudini
            for n in range(n_iterations): #ciclo sulle iterazioni
                cell_grid_iteration[n] = sample_mean[n][0][i][j] #n-esima iterazione, plev fissato, i-esimo elemento lat, primo elemento lon
            #Fuori dalle iterazioni perché ragiono sulla distribuzione, ottenuta dopo tutte le iterazioni
            # media e la deviazione standard
            array_mean[i,j] = np.mean(cell_grid_iteration)
            array_std[i,j] = np.std(cell_grid_iteration, ddof=1)  # Specifica ddof=1 per calcolare la deviazione standard campionaria (ddof = 1 --> divisione per N-1)
            # quinto e il 95-esimo percentile
            array_2th_percentile[i,j] = np.percentile(cell_grid_iteration, 2.5)
            array_97th_percentile[i,j] = np.percentile(cell_grid_iteration, 97.5)
    return array_mean,array_std,array_2th_percentile,array_97th_percentile

#Funzione che fa il plot di media, std, 2.5th e 97.5th percentile della distribuzione bootstrap
def plot_bs_95cl_mean_std(name_list,name_dict,array_mean,array_std,array_2th_percentile,array_97th_percentile):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,8), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots

    #valori estremanti di lon-lat
    min_lon = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lon.min()
    max_lon = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lon.max()
    min_lat = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lat.min()
    max_lat = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lat.max()

    ax[0,0].imshow(array_2th_percentile[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[0,0].invert_yaxis()
    ax[0,0].set_title('2.5th percentile')
    fig.colorbar(ax[0,0].imshow(array_2th_percentile, cmap='seismic'), ax=ax[0,0])

    ax[0,1].imshow(array_97th_percentile[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[0,1].invert_yaxis()
    ax[0,1].set_title('97.5th percentile')
    fig.colorbar(ax[0,1].imshow(array_97th_percentile, cmap='seismic'), ax=ax[0,1])

    ax[1,0].imshow(array_mean[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[1,0].invert_yaxis()
    ax[1,0].set_title('Mean of bootstrap distribution')
    fig.colorbar(ax[1,0].imshow(array_mean, cmap='seismic'), ax=ax[1,0])

    ax[1,1].imshow(array_std[::-1], cmap='Reds', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[1,1].invert_yaxis()
    ax[1,1].set_title('Std of bootstrap distribution')
    fig.colorbar(ax[1,1].imshow(array_std, cmap='Reds'), ax=ax[1,1])

    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlabel('lon')
            ax[i,j].set_ylabel('lat')
            ax[i,j].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax[i,j].coastlines()        
            ax[i,j].set_xticks(np.arange(min_lon,max_lon, 20))
            ax[i,j].set_yticks(np.arange(min_lat,max_lat, 10))
    fig.suptitle('Bootstrap distribution with 4 models')
    fig.show()

#funzione che calcola matrix10
def bs_compute_matrix10(name_list,name_dict,array_2th_percentile,array_97th_percentile): #funzione che calcola la matrice di 1 e 0 di dimensioni (30,78), che ha 1 dove l'elemento ij-esimo della matrice mean_bias è <=5th percentile oppure >=95th percentile e 0 altrimenti
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_list)
    #mean_bias[0], array_2.5th_percentile, array_97.5th_percentile --> matrici (30,78)
    #Se l'elemento ij-esimo di mean_bias è <=2.5th percentile oppure >=97.5th percentile allora elemento ij-esimo è statisticamente differente --> metto un 1 nella matrice che creo
    #Creo la matrice che ha 1 nei pti statisticamente differenti e 0 nei pti che non lo sono
    matrix10 = np.zeros((len(mean_bias[0].lat.values),len(mean_bias[0].lon.values))) #inizializzo la matrice di 1 e 0, di dimensioni (30,78)
    for i in range(len(mean_bias[0].lat.values)): #ciclo sulle latitudini
        for j in range(len(mean_bias[0].lon.values)): #ciclo sulle longitudini
            if mean_bias[0][i][j] <= array_2th_percentile[i,j] or mean_bias[0][i][j] >= array_97th_percentile[i,j]: #<=2.5th oppure >=97.5th percentile --> statisticamente differenti
                matrix10[i,j] = 1
            else:
                matrix10[i,j] = 0 #superfluo
    return matrix10

#funzione per il plot del cluster medio + i puntini di significatività
def plot_bs_mean_cluster_matrix10(name_list,name_dict,fig_size,v_min,v_max,matrix10,title_plot):#funzione che plotta il cluster medio + i pti significativamente differenti dalla distribuzione bootstrap
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_list)
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()})
    #plot
    plot_mod = ax.pcolormesh(mean_bias[0].lon, mean_bias[0].lat, mean_bias[0], vmin=v_min, vmax=v_max, cmap='seismic')
    coords = np.where(matrix10 == 1) #array di valori di longitudini e latitudini in cui matrix10 = 1
    # Plot dei punti solo dove matrix10 è uguale a 1
    ax.plot(mean_bias[0].lon[coords[1]], mean_bias[0].lat[coords[0]], marker='o', color='black', markersize=2, linestyle='None', transform=ccrs.PlateCarree())
    #imposto lat-lon sugli assi
    ax.set_xticks(np.arange(mean_bias[0].lon.min(), mean_bias[0].lon.max(), 20))
    ax.set_yticks(np.arange(mean_bias[0].lat.min(), mean_bias[0].lat.max(), 10))
    #aggiungo coste
    ax.coastlines()
    #Label assi
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    #barra di colori
    fig.colorbar(plot_mod,ax=ax)
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.show()

#plot di diff, cioè la differenza tra il cluster medio e la media della distribuzione bootstrap --> in più ci metto anche i punti di significatività
def plot_bs_diff_cluster(diff,title_plot,v_min,v_max,fig_size,matrix10): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli da plottare, name_dict è o models_atmos
    fig, ax = plt.subplots(figsize=fig_size,subplot_kw={"projection": ccrs.PlateCarree()}) #trasformazione cartografica = lonxlat   
    plot_mod = ax.pcolormesh(diff[0].lon, diff[0].lat, diff[0],vmin=v_min, vmax=v_max,cmap='seismic') 
    coords = np.where(matrix10 == 1) #array di valori di longitudini e latitudini in cui matrix10 = 1
    # Plot dei punti solo dove matrix10 è uguale a 1
    ax.plot(diff[0].lon[coords[1]], diff[0].lat[coords[0]], marker='o', color='black', markersize=2, linestyle='None', transform=ccrs.PlateCarree())
    #imposto lat-lon sugli assi
    #valori assi            
    ax.set_xticks(np.arange(diff[0].lon.min(),diff[0].lon.max(), 20))
    ax.set_yticks(np.arange(diff[0].lat.min(),diff[0].lat.max(), 10))
    ax.coastlines() #gca = get current axis
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')
    # Titolo
    fig.colorbar(plot_mod)
    fig.suptitle(title_plot, fontsize=16, y=1.02)

#zonmean
#funzione che estrae in modo random un numero di modelli e ne calcola il valor medio di bias DJF --> per un singolo cluster
def bs_sample_mean_zonmean(n_iterations,name_dict,name_list): 
    #creazione di una lista con il nome dei modelli
    list_name_models_atmos = list(name_dict.keys())
    sample_rand = [] #inizializzazione lista di modelli presi in modo random, in numero n_iterations
    sample_mean = [] #inizializzazione di una lista in cui vado ad inserire la media di ogni sample, preso con l'estrazione random
    for n in range(n_iterations): #itero n_iterations volte
        sample_rand = random.sample(range(len(name_dict)), len(name_list)) #lista = estraggo 4 numeri random da 0 a 36, che sono il numero associato ad ogni modello
        sample_sum = 0 #inizializzazione ad ogni iterazione di sample_sum
        for i in range(len(name_list)): #ciclo sui 4 modelli presi
            sample_sum = sample_sum + name_dict[list_name_models_atmos[sample_rand[i]]]['zonmean bias DJF']
        sample_mean.append(sample_sum/len(sample_rand)) #calcolo il valore medio per ogni cluster
    return sample_mean

#funzione che calcola l'array mean-std-5th-95th della distribuzione bootstrap per ogni grid cell
def bs_compute_array_mean_std_95cl_zonmean(n_iterations,sample_mean):
    #Inizializzazione array
    cell_grid_iteration = np.zeros(n_iterations) #array in cui metto i valori medi di un solo punto griglia per iterazioni diverse
    array_mean = np.zeros((len(sample_mean[0].plev),len(sample_mean[0].lat))) #array che racchiude i valori medi delle distribuzioni per ogni punto griglia
    array_std = np.zeros((len(sample_mean[0].plev),len(sample_mean[0].lat))) #array che racchiude le deviazioni standard delle distribuzioni per ogni punto griglia
    array_2th_percentile = np.zeros((len(sample_mean[0].plev),len(sample_mean[0].lat))) #array che racchiude i valori in cui si ha il 5th percentile delle distribuzioni per ogni pt griglia
    array_97th_percentile = np.zeros((len(sample_mean[0].plev),len(sample_mean[0].lat))) #array che racchiude i valori in cui si ha il 95th percentile delle distribuzioni per ogni pt griglia
    #Determino mean, std, 5th-95th percentile delle distribuzioni per ogni pt griglia
    for i in range(len(sample_mean[0].plev)): #ciclo su plev
        for j in range(len(sample_mean[0].lat)): #ciclo sulle latitudinni
            for n in range(n_iterations): #ciclo sulle iterazioni
                cell_grid_iteration[n] = sample_mean[n][i][j][0] #n-esima iterazione, plev fissato, i-esimo elemento lat, primo elemento lon
                #print(n,i,j)
            #Fuori dalle iterazioni perché ragiono sulla distribuzione, ottenuta dopo tutte le iterazioni
            # media e la deviazione standard
            array_mean[i,j] = np.mean(cell_grid_iteration)
            array_std[i,j] = np.std(cell_grid_iteration, ddof=1)  # Specifica ddof=1 per calcolare la deviazione standard campionaria (ddof = 1 --> divisione per N-1)
            # quinto e il 95-esimo percentile
            array_2th_percentile[i,j] = np.percentile(cell_grid_iteration, 2.5)
            array_97th_percentile[i,j] = np.percentile(cell_grid_iteration, 97.5)
    return array_mean,array_std,array_2th_percentile,array_97th_percentile

#Funzione che fa il plot di media, std, 5th e 95th percentile della distribuzione bootstrap
def plot_bs_95cl_mean_std_zonmean(name_list,name_dict,array_mean,array_std,array_2th_percentile,array_97th_percentile):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots

    #valori estremanti di plev-lat
    min_plev = name_dict[name_list[0]]['zonmean bias DJF'].plev.min().values
    max_plev = name_dict[name_list[0]]['zonmean bias DJF'].plev.max().values
    min_lat = name_dict[name_list[0]]['zonmean bias DJF'].lat.min().values
    max_lat = name_dict[name_list[0]]['zonmean bias DJF'].lat.max().values

    ax[0,0].imshow(array_2th_percentile[::-1], cmap='seismic', extent=[min_lat, max_lat, min_plev, max_plev])
    ax[0,0].set_title('2.5th percentile')
    fig.colorbar(ax[0,0].imshow(array_2th_percentile, cmap='seismic'), ax=ax[0,0])

    ax[0,1].imshow(array_97th_percentile[::-1], cmap='seismic', extent=[min_lat, max_lat, min_plev, max_plev])
    ax[0,1].set_title('97.5th percentile')
    fig.colorbar(ax[0,1].imshow(array_97th_percentile, cmap='seismic'), ax=ax[0,1])

    ax[1,0].imshow(array_mean[::-1], cmap='seismic', extent=[min_lat, max_lat, min_plev, max_plev])
    ax[1,0].set_title('Mean of bootstrap distribution')
    fig.colorbar(ax[1,0].imshow(array_mean, cmap='seismic'), ax=ax[1,0])

    ax[1,1].imshow(array_std[::-1], cmap='Reds', extent=[min_lat, max_lat, min_plev, max_plev])
    ax[1,1].set_title('Std of bootstrap distribution')
    fig.colorbar(ax[1,1].imshow(array_std, cmap='Reds'), ax=ax[1,1])

    #invert y axis and set ticks on x-y axes
    for i in range(2):
        for j in range(2):
            ax[i,j].invert_yaxis()
            ax[i,j].set_xlabel('lat')
            ax[i,j].set_ylabel('plev')
            #ax[i,j].set_xlim(min_lat, max_lat)
            #ax[i,j].set_ylim(min_plev, max_plev)      
            #ax[i,j].set_yticks(np.arange(min_plev, max_plev, 7500))
            #ax[i,j].set_xticks(np.arange(min_lat,max_lat, 10))
    fig.suptitle('Bootstrap distribution')
    fig.show()

def bs_compute_matrix10_zonmean(name_list,name_dict,array_2th_percentile,array_97th_percentile): #funzione che calcola la matrice di 1 e 0 di dimensioni (30,78), che ha 1 dove l'elemento ij-esimo della matrice mean_bias è <=5th percentile oppure >=95th percentile e 0 altrimenti
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        zonmean = name_dict[model_name]['zonmean bias DJF']
        zonmean = zonmean.assign_coords({"plev" : zonmean.plev.round()}) #arrotondo in modo tale che i livelli di pressione siano gli stessi per ogni modello
        sum_bias = sum_bias + zonmean
    #valor medio
    mean_bias = sum_bias / len(name_list)
    #Se l'elemento ij-esimo di mean_bias è <=5th percentile oppure >=95th percentile allora elemento ij-esimo è statisticamente differente --> metto un 1 nella matrice che creo
    #Creo la matrice che ha 1 nei pti statisticamente differenti e 0 nei pti che non lo sono
    matrix10 = np.zeros((len(mean_bias.plev.values),len(mean_bias.lat.values))) #inizializzo la matrice di 1 e 0, di dimensioni (plev,lat)
    for i in range(len(mean_bias.plev.values)): #ciclo su plev
        for j in range(len(mean_bias.lat.values)): #ciclo sulle latitudini
            if mean_bias[i,j,0] <= array_2th_percentile[i,j] or mean_bias[i,j,0] >= array_97th_percentile[i,j]: #<=5th oppure >=95th percentile --> statisticamente differenti
                matrix10[i,j] = 1
            else:
                matrix10[i,j] = 0 #superfluo
    return matrix10

#funzione per il plot del cluster medio + i puntini di significatività
def plot_bs_mean_cluster_matrix10_zonmean(name_list,name_dict,fig_size,v_min,v_max,matrix10,title_plot):#funzione che plotta il cluster medio + i pti significativamente differenti dalla distribuzione bootstrap
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        zonmean = name_dict[model_name]['zonmean bias DJF']
        zonmean = zonmean.assign_coords({"plev" : zonmean.plev.round()}) #arrotondo in modo tale che i livelli di pressione siano gli stessi per ogni modello
        sum_bias = sum_bias + zonmean
    #valor medio
    mean_bias = sum_bias / len(name_list)
    fig, ax = plt.subplots(figsize=fig_size)
    #plot
    plot_mod = ax.pcolormesh(mean_bias.lat, mean_bias.plev, mean_bias[:,:,0], vmin=v_min, vmax=v_max, cmap='seismic')
    coords = np.where(matrix10 == 1) #array di valori di longitudini e latitudini in cui matrix10 = 1
    # Plot dei punti solo dove matrix10 è uguale a 1
    ax.plot(mean_bias.lat[coords[1]], mean_bias.plev[coords[0]], marker='o', color='black', markersize=2, linestyle='None')
    #imposto lat-lon sugli assi
    #ax.set_yticks(np.arange(mean_bias.plev.min(), mean_bias.plev.max(), 7500))
    #ax.set_xticks(np.arange(mean_bias.lat.min(), mean_bias.lat.max(), 10))
    ax.invert_yaxis()
    #Label assi
    ax.set_ylabel('plev')
    ax.set_xlabel('lat')
    #barra di colori
    fig.colorbar(plot_mod,ax=ax)
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.show()

def plot_bs_diff_cluster_zonmean(diff,title_plot,v_min,v_max,fig_size,matrix10): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli da plottare, name_dict è o models_zonmean
    fig, ax = plt.subplots(figsize=fig_size) #trasformazione cartografica = lonxlat   
    plot_mod = ax.pcolormesh(diff.lat, diff.plev, diff, cmap='seismic', vmin=v_min, vmax=v_max)
    coords = np.where(matrix10 == 1) #array di valori di longitudini e latitudini in cui matrix10 = 1
    # Plot dei punti solo dove matrix10 è uguale a 1
    ax.plot(diff.lat[coords[1]], diff.plev[coords[0]], marker='o', color='black', markersize=2, linestyle='None')
    ax.invert_yaxis()
    #label assi            
    ax.set_ylabel('plev')
    ax.set_xlabel('latitude')
    # Titolo
    fig.colorbar(plot_mod)
    fig.suptitle(title_plot, fontsize=16, y=1.02)

#EGR --> funzioni per il calcolo di Eady Growth Rate
# 1.
#funzione che per ogni modello ritorna il valore di theta
def compute_theta(name_dict,name_model): #name_dict = models_ta, name_model = 'TaiESM1',...
    model = name_dict[name_model]['ta North Atlantic box'] # dimensioni (time,plev,lat,lon)
    #Costanti
    R_cp = 0.286 # R / cp
    p0 = 1.013e5 #1000 hPa
    #Inizializzo theta come un xarray che ha dimensioni = (time,plev,lat,lon)
    theta = xr.DataArray(np.empty(model['ta'].shape), dims=model['ta'].dims, coords=model['ta'].coords)
    #Calcolo theta
    for i in range(len(model.plev.values)): #ciclo sui plev, cioè in verticale
        p = model.plev[i].values #livello i-esimo di pressione
        theta[:,i,:,:] = model['ta'][:,i,:,:] * ((p0 / p) ** R_cp) #lon=0
    return theta 
#2
#funzione che calcola la derivata di theta e ua rispetto a z
def compute_derivative(name_variable,temperature): # name_variable = theta or ua, name_variable = nome della variabile di cui si vuole calcolare la derivata (dizionario completo), temperature=dizionario in cui è riportata la temperatura (no seas mean)
    #Inizializzo derivata della variabile come un xarray
    derivative = xr.DataArray(np.empty(name_variable[:,1:-1,:,:].shape), dims=name_variable[:,1:-1,:,:].dims, coords=name_variable[:,1:-1,:,:].coords)
    #La derivata la devo fare rispetto a z, non rispetto a p --> ricavo z
    z = np.zeros(len(name_variable.plev)) #array di lunghezza pari al numero di pressioni
    p0 = 1.013e5 #pressione a 1000hPa
    #rho_0 = 1.29 #densità aria in [kg/m^3] alla pressione p0
    #g = 9.81 #accelerazione di gravità (m/s^2)
    for i in range(len(name_variable.plev)):
        H = 29.3*temperature[:,i,:,:].mean() #altezza scala, utilizzo il campo di temperatura assoluta e poi lo medio rispetto al tempo, alla latitudine e alla longitudine --> rimane un campo di temperatura dipendente esclusivamente da plev
        #z[i] = -(p0/(rho_0*g))*(math.log(name_variable.plev[i].values/p0))
        z[i] = -H*(math.log(name_variable.plev[i].values/p0))
    for i in range(len(name_variable.plev)-2): #ciclo sui plev - 2
        derivative[:,i,:,:] = (name_variable[:,i,:,:] - name_variable[:,i+2,:,:]) / (z[i] - z[i+2])
    return derivative
#3.
#Funzione che calcola la frequenza di Brunt–Väisälä
def compute_frequency(theta,derivative_theta): #dove theta e derivative_theta sono i due dizionari models_ta[name]['potential... oppure theta derivative DJF]
    g = 9.81 #accelerazione di gravità (m/s^2)
    #Inizializzo N^2 e N come xarray
    N = xr.DataArray(np.empty(theta[:,1:-1,:,:].shape), dims=theta[:,1:-1,:,:].dims, coords=theta[:,1:-1,:,:].coords)
    N_quadro = xr.DataArray(np.empty(theta[:,1:-1,:,:].shape), dims=theta[:,1:-1,:,:].dims, coords=theta[:,1:-1,:,:].coords)
    for i in range(len(theta.plev)-2): #ciclo su plev
        N_quadro[:,i,:,:] = (g/abs(theta[:,i+1,:,:]))*abs(derivative_theta[:,i,:,:]) #theta[i+1,...] perché il livello plev=1000hPa non c'è nella derivata di theta
        N[:,i,:,:] = np.where(N_quadro[:,i,:,:] < 0, np.nan,np.sqrt(N_quadro[:,i,:,:])) #gli elementi negativi vengono messi pari a nan (vuol dire che dtheta/dz < 0 e quindi profilo instabile), degli altri si calcola la radice quadrata
    return N
#4.
#Funzione che calcola il parametro di Coriolis f = 2*omega*sin(phi)
def compute_coriolis_parameter(name_variable): #f come un array 1d di dimensioni pari al numero di lat.values, perché f è dipendente solo dalla lat phi
    omega = 7.2921e-5 # rad/s
    #inizializzo f come un xarray
    f = xr.DataArray(np.empty(name_variable[0,0,:,0].shape), dims=name_variable[0,0,:,0].dims, coords=name_variable[0,0,:,0].coords)
    lat_rad = np.deg2rad(name_variable.lat.values) #converto in radianti i valori di latitudine
    # Calcoloil seno delle latitudini
    seno = np.sin(lat_rad)
    f[:] = 2 * omega * seno
    return f

#5.
#Funzione che calcola Eady Growth Rate
def compute_egr(f,derivative_u,N):
    c = 0.3068
    #Inizializzo sigma
    sigma = xr.DataArray(np.empty(N[:,:,:,:].shape), dims=N[:,:,:,:].dims, coords=N[:,:,:,:].coords)
    #Calcolo EGR
    for i in range(len(N.plev)): #ciclo sui plev
        for j in range(len(N.lat)): #ciclo su lat
            sigma[:,i,j,:] = (c * f[j] * abs(derivative_u[:,i,j,:])) / N[:,i,j,:]
    return sigma

# 1.
#funzione che per ogni modello ritorna il valore di theta per DJF
def compute_theta_era(dataset): 
    #Costanti
    R_cp = 0.286 # R / cp
    p0 = 1.013e5 #1000 hPa
    #Inizializzo theta come un array che ha dimensioni = (plev,lat,lon)
    theta = xr.DataArray(np.empty(dataset[:,:,:,:].shape), dims=dataset[:,:,:,:].dims, coords=dataset[:,:,:,:].coords)
    #Calcolo theta
    for i in range(len(dataset.plev.values)): #ciclo sui plev, cioè in verticale
        p = dataset.plev[i].values #livello i-esimo di pressione
        theta[:,i,:,:] = dataset[:,i,:,:] * ((p0 / p) ** R_cp) #lon=0
    return theta  

#BOOTSTRAP
