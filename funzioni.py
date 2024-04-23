import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import random
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
    #plot del valor medio per il cluster 3
    plot_mod = mean_bias.plot.pcolormesh(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='seismic',
    subplot_kws={"projection":ccrs.PlateCarree()})  #trasformazione cartografica = lonxlat
    plt.gca().coastlines() #gca = get current axis
    #valori assi            
    #plt.xticks(np.arange(mean_bias[0].lon.min(),mean_bias[0].lon.max(), 20))
    #plt.yticks(np.arange(mean_bias[0].lat.min(),mean_bias[0].lat.max(), 10))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

#plot della standard deviation dei cluster di tos
def plot_std_cluster_tos(name_models_to_plot,name_dict,v_min,v_max,title_plot,title_pdf): #name_dict è models_tos
    dataset = [] #Inizializzo una lista
    for i in range(len(name_models_to_plot)): #Vado ad inserire all'interno di dataset tutti gli elementi 'North Atlantic bias DJF' della j-esima lista, dove j = 0,...,4
        dataset.append(name_dict[name_models_to_plot[i]]['North Atlantic bias DJF'])

    combined_data = xr.concat(dataset, dim='time') #concateno tutti gli elementi all'interno di dataset, lungo la dimensione time
    std_dev = combined_data.std(dim='time') #calcolo la deviazione standard lungo la dimensione time
    # Plot
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max,
    subplot_kws={"projection":ccrs.PlateCarree()})  #trasformazione cartografica = lonxlat    
    #valori assi            
    #plt.xticks(np.arange(combined_data[0].lon.min(),combined_data[0].lon.max(), 20))
    #plt.yticks(np.arange(combined_data[0].lat.min(),combined_data[0].lat.max(), 10))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.gca().coastlines() #gca = get current axis
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

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
            plot_mod = ax[i,j].pcolormesh(data_array[0].lon, data_array[0].lat, data_array[0], cmap='seismic', vmin=v_min, vmax=v_max)
            #Plot della climatologia dei singoli mdoelli e di ERA5
            data = name_dict[model_name]['atmos North Atlantic seasonal mean DJF']     
            data_era = dataset_seas_mean[4]
            #plot
            data[0].plot.contour(ax=ax[i,j],colors='k')
            data_era[0].plot.contour(ax=ax[i,j],colors='g')
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
        data[0].plot.contour(ax=ax[i],colors='k')
        data_era[0].plot.contour(ax=ax[i],colors='g')
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

    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40,)
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#plot dei cluster medi atmos
def plot_mean_cluster_atmos(name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf,v_min,v_max,fig_size): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli da plottare, name_dict è o models_atmos
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_models_to_plot)):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_models_to_plot)
    #plot del valor medio per il cluster 3
    plot_mod = mean_bias[0].plot.pcolormesh(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='seismic',
    subplot_kws={"projection":ccrs.PlateCarree()})  #trasformazione cartografica = lonxlat   
    data_era = dataset_seas_mean[4]
    #plot
    data_era[0].plot.contour(colors='g')
    #valori assi            
    plt.xticks(np.arange(mean_bias[0].lon.min(),mean_bias[0].lon.max(), 20))
    plt.yticks(np.arange(mean_bias[0].lat.min(),mean_bias[0].lat.max(), 10))
    plt.gca().coastlines() #gca = get current axis
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

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
    plt.xticks(np.arange(combined_data[0].lon.min(),combined_data[0].lon.max(), 20))
    plt.yticks(np.arange(combined_data[0].lat.min(),combined_data[0].lat.max(), 10))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.gca().coastlines() #gca = get current axis)
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

#ZONAVG
#plot medie zonali
def plot_zonavg(n_rows, n_cols, fig_size, name_models_to_plot, name_dict, dataset_seas_mean, v_min, v_max, title_plot, title_pdf):
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
            data_array = name_dict[model_name]['zonavg bias DJF']
            plot_mod = ax[i, j].pcolormesh(data_array.lat, data_array.plev, data_array.sel(lon=0), cmap='seismic', vmin=v_min, vmax=v_max)
            # Plot della climatologia dei singoli modelli e di ERA5
            data = name_dict[model_name]['zonavg seasonal mean DJF']
            data_era = dataset_seas_mean[4]
            # Plot
            data.sel(lon=0).plot.contour(ax=ax[i, j], colors='k')  # In modo che l'array sia 2D su plev e lat
            data_era.sel(lon=0).plot.contour(ax=ax[i, j], colors='g')
            ax[i, j].set_ylabel('plev')
            ax[i, j].set_xlabel('lat')
            ax[i, j].set_title(model_name)  # Nome di ogni singolo modello sul plot
            # Inverto l'asse y in modo che i livelli di pressione siano corretti
            ax[i, j].invert_yaxis()

    # Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= len(model_name):
                ax[i, j].axis('off')

    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.colorbar(plot_mod, ax=ax, orientation='horizontal', shrink=0.6, aspect=40)
    fig.savefig(title_pdf, format='pdf')

#plot medie zonali per due cluster
def plot_zonavg_2_cluster(fig_size,name_models_to_plot,name_dict,dataset_seas_mean,v_min,v_max,title_plot,title_pdf):
    #plot medie annuali dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    # Plot dei modelli
    for i in range(2): #ciclo sulle colonne
        model_name = name_models_to_plot[i]
        data_array = name_dict[model_name]['zonavg bias DJF']
        plot_mod = ax[i].pcolormesh(data_array.lat, data_array.plev, data_array.sel(lon=0), cmap='seismic', vmin=v_min, vmax=v_max)     
        #Plot della climatologia dei singoli mdoelli e di ERA5
        data = name_dict[model_name]['zonavg seasonal mean DJF']        
        data_era = dataset_seas_mean[4]
        #plot
        data.sel(lon=0).plot.contour(ax=ax[i],colors='k')
        data_era.sel(lon=0).plot.contour(ax=ax[i],colors='g')
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
def plot_mean_cluster_zonavg(number_models,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf,v_min,v_max,fig_size):
    #Inizializzo sum_zonavg per il calcolo della media
    sum_zonavg = 0
    #calcolo il valor medio
    for i in range(number_models):
        model_name = name_models_to_plot[i]
        zonavg = name_dict[model_name]['zonavg bias DJF']
        zonavg = zonavg.assign_coords({"plev" : zonavg.plev.round()}) #arrotondo in modo tale che i livelli di pressione siano gli stessi per ogni modello
        sum_zonavg = sum_zonavg + zonavg
    #valor medio
    mean_zonavg = sum_zonavg / number_models
    #plot del valor medio
    mean_zonavg.plot(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='seismic') 
    data_era = dataset_seas_mean[4]
    # Plot
    data_era.sel(lon=0).plot.contour(colors='g')
    plt.ylabel('plev')
    plt.xlabel('latitude')
    plt.gca().invert_yaxis()
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

#plot della standard deviation dei cluster di zonavg
def plot_std_cluster_zonavg(name_models_to_plot,name_dict,v_min,v_max,title_plot,title_pdf): #name_dict è models_zonavg
    dataset = [] #Inizializzo una lista
    for i in range(len(name_models_to_plot)): #Vado ad inserire all'interno di dataset tutti gli elementi 'zonavg bias DJF' della j-esima lista, dove j = 0,...,4
        dataset.append(name_dict[name_models_to_plot[i]]['zonavg bias DJF'])

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

    #Bootstrap
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

#funzione che calcola l'array mean-std-5th-95th della distribuzione bootstrap per ogni grid cell
def bs_compute_array_mean_std_5th_95th(n_iterations,sample_mean):
    #Inizializzazione array
    cell_grid_iteration = np.zeros(n_iterations) #array in cui metto i valori medi di un solo punto griglia per iterazioni diverse
    array_mean = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori medi delle distribuzioni per ogni punto griglia
    array_std = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude le deviazioni standard delle distribuzioni per ogni punto griglia
    array_5th_percentile = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori in cui si ha il 5th percentile delle distribuzioni per ogni pt griglia
    array_95th_percentile = np.zeros((len(sample_mean[0].lat),len(sample_mean[0].lon))) #array che racchiude i valori in cui si ha il 95th percentile delle distribuzioni per ogni pt griglia
    #Determino mean, std, 5th-95th percentile delle distribuzioni per ogni pt griglia
    for i in range(len(sample_mean[0].lat)): #ciclo sulle latitudini
        for j in range(len(sample_mean[0].lon)): #ciclo sulle longitudini
            for n in range(n_iterations): #ciclo sulle iterazioni
                cell_grid_iteration[n] = sample_mean[n][0][i][j] #n-esima iterazione, plev fissato, i-esimo elemento lat, primo elemento lon
            #Fuori dalle iterazioni perché ragiono sulla distribuzione, ottenuta dopo tutte le iterazioni
            # media e la deviazione standard
            array_mean[i,j] = np.mean(cell_grid_iteration)
            array_std[i,j] = np.std(cell_grid_iteration, ddof=1)  # Specifica ddof=1 per calcolare la deviazione standard campionaria
            # quinto e il 95-esimo percentile
            array_5th_percentile[i,j] = np.percentile(cell_grid_iteration, 5)
            array_95th_percentile[i,j] = np.percentile(cell_grid_iteration, 95)
    return array_mean,array_std,array_5th_percentile,array_95th_percentile

#Funzione che calcola la media, la std, il 5th e 95th percentile della distribuzione bootstrap
def plot_bs_5th_95th_mean_std(name_list,name_dict,array_mean,array_std,array_5th_percentile,array_95th_percentile):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,8), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots

    #valori estremanti di lon-lat
    min_lon = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lon.min()
    max_lon = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lon.max()
    min_lat = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lat.min()
    max_lat = name_dict[name_list[0]]['atmos North Atlantic bias DJF'].lat.max()

    ax[0,0].imshow(array_5th_percentile[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[0,0].invert_yaxis()
    ax[0,0].set_title('5th percentile')
    fig.colorbar(ax[0,0].imshow(array_5th_percentile, cmap='seismic'), ax=ax[0,0])

    ax[0,1].imshow(array_95th_percentile[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[0,1].invert_yaxis()
    ax[0,1].set_title('95th percentile')
    fig.colorbar(ax[0,1].imshow(array_95th_percentile, cmap='seismic'), ax=ax[0,1])

    ax[1,0].imshow(array_mean[::-1], cmap='seismic', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[1,0].invert_yaxis()
    ax[1,0].set_title('Mean of bootstrap distribution')
    fig.colorbar(ax[1,0].imshow(array_mean, cmap='seismic'), ax=ax[1,0])

    ax[1,1].imshow(array_std[::-1], cmap='Reds', extent=[min_lon, max_lon, min_lat, max_lat], transform=ccrs.PlateCarree())
    ax[1,1].invert_yaxis()
    ax[1,1].set_title('Std of bootstrap distribution')
    fig.colorbar(ax[1,1].imshow(array_std, cmap='Reds'), ax=ax[1,1])

    #invert y axis and set ticks on x-y axes
    for i in range(2):
        for j in range(2):
            #ax[i,j].invert_yaxis()
            #ax[i,j].set_xlim(mean_bias[0].lon.min(),mean_bias[0].lon.max())
            #ax[i,j].set_ylim(mean_bias[0].lat.min(),mean_bias[0].lat.max())
            ax[i,j].set_xlabel('lon')
            ax[i,j].set_ylabel('lat')
            ax[i,j].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax[i,j].coastlines()        
            ax[i,j].set_xticks(np.arange(min_lon,max_lon, 20))
            ax[i,j].set_yticks(np.arange(min_lat,max_lat, 10))
    fig.suptitle('Bootstrap distribution with 4 models')
    fig.show()

#funzione che calcola matrix10
def bs_compute_matrix10(name_list,name_dict,array_5th_percentile,array_95th_percentile): #funzione che calcola la matrice di 1 e 0 di dimensioni (30,78), che ha 1 dove l'elemento ij-esimo della matrice mean_bias è <=5th percentile oppure >=95th percentile e 0 altrimenti
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_list)):
        model_name = name_list[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_list)
    #mean_bias[0], array_5th_percentile, array_95th_percentile --> matrici (30,78)
    #Se l'elemento ij-esimo di mean_bias è <=5th percentile oppure >=95th percentile allora elemento ij-esimo è statisticamente differente --> metto un 1 nella matrice che creo
    #Creo la matrice che ha 1 nei pti statisticamente differenti e 0 nei pti che non lo sono
    matrix10 = np.zeros((len(mean_bias[0].lat.values),len(mean_bias[0].lon.values))) #inizializzo la matrice di 1 e 0, di dimensioni (30,78)
    for i in range(len(mean_bias[0].lat.values)): #ciclo sulle latitudini
        for j in range(len(mean_bias[0].lon.values)): #ciclo sulle longitudini
            if mean_bias[0][i][j] <= array_5th_percentile[i,j] or mean_bias[0][i][j] >= array_95th_percentile[i,j]: #<=5th oppure >=95th percentile --> statisticamente differenti
                matrix10[i,j] = 1
            else:
                matrix10[i,j] = 0
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
