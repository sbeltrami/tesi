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
def plot_bias_tos(n_rows,n_cols,fig_size,number_models,name_models_to_plot,name_dict,val_min,val_max,title_plot,title_pdf): #number_models è la lista che riporta il numero di modelli in un determinato cluster
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
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
            ax[i,j].set_title(model_name) #nome di ogni singolo modello sul plot

    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= number_models:
                ax[i, j].axis('off')
    
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)

    fig.savefig(title_pdf, format='pdf')

#plot dei bias dei 2 modelli
def plot_bias_2_models_tos(fig_size,val_min,val_max,name_models_to_plot,name_dict,title_plot,title_pdf):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 righe
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        plot_mod = name_dict[model_name]['North Atlantic bias DJF'].plot.pcolormesh(ax=ax[i])
         # Fisso la scala
        plot_mod.set_clim(vmin=val_min, vmax=val_max)
        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')
        ax[i].set_title(model_name) #nome di ogni singolo modello sul plot


    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')
    
    
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
    plot_mod = mean_bias.plot.pcolormesh(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='RdBu')
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
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max)
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

#ATMOS
#funzione per plot bias atmos
def plot_bias_atmos(n_rows,n_cols,fig_size,v_min,v_max,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli, name_dict è model_atmos, val_min e max sono i valori che fissano la scala, dataset_seas_mean è era_na_seas_mean
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots

    for i in range(n_rows): #ciclo sulle righe
        for j in range(n_cols): #ciclo sulle colonne
            models_index_list = i * n_cols + j #indice del modello all'interno della lista
            if models_index_list == len(name_models_to_plot):
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['atmos North Atlantic bias DJF']
            plot_mod = data_array[0].plot.pcolormesh(ax=ax[i, j])
            # Fisso la scala
            plot_mod.set_clim(vmin=v_min, vmax=v_max)

            #Plot della climatologia dei singoli mdoelli e di ERA5
            data = name_dict[model_name]['atmos North Atlantic seasonal mean DJF']     
            data_era = dataset_seas_mean[4]
            #plot
            data[0].plot.contour(ax=ax[i,j],colors='k')
            data_era[0].plot.contour(ax=ax[i,j],colors='g')
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

    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)

    fig.savefig(title_pdf, format='pdf')

#plot del bias per due modelli
def plot_bias_2_models_atmos(fig_size,v_min,v_max,name_models_to_plot,name_dict,dataset_seas_mean,title_plot,title_pdf):
    #Plot dei modelli
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)  # Modificato per 2 righe e 1 colonna

    # Plot dei modelli
    for i in range(2):  # Solo 2 colonne
        model_name = name_models_to_plot[i]  # Usa l'indice i direttamente
        data_array = name_dict[model_name]['atmos North Atlantic bias DJF']
        plot_mod = data_array[0].plot.pcolormesh(ax=ax[i])
         # Fisso la scala
        plot_mod.set_clim(vmin=v_min, vmax=v_max)

        #Plot della climatologia dei singoli mdoelli e di ERA5
        data = name_dict[model_name]['atmos North Atlantic seasonal mean DJF']        
        data_era = dataset_seas_mean[4]
        #plot
        data[0].plot.contour(ax=ax[i],colors='k')
        data_era[0].plot.contour(ax=ax[i],colors='g')
        #label assi
        ax[i].set_ylabel('latitude')
        ax[i].set_xlabel('longitude')
        ax[i].set_title(model_name) #nome di ogni singolo modello sul plot

    # Rimuovi i quadrati non utilizzati
    for i in range(2):
        if i >= len(name_models_to_plot):  # Modificato per usare solo l'indice i
            ax[i].axis('off')    
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)

    fig.savefig(title_pdf, format='pdf')


#plot dei cluster medi atmos
def plot_mean_cluster_atmos(name_models_to_plot,name_dict,title_plot,title_pdf,v_min,v_max,fig_size): #name_models_to_plot indica la lista in cui sono racchiusi i nomi dei modelli da plottare, name_dict è o models_atmos
    #Inizializzo sum_bias per il calcolo della media
    sum_bias = 0
    #calcolo il valor medio
    for i in range(len(name_models_to_plot)):
        model_name = name_models_to_plot[i]
        sum_bias = sum_bias + name_dict[model_name]['atmos North Atlantic bias DJF']
    #valor medio
    mean_bias = sum_bias / len(name_models_to_plot)
    #plot del valor medio per il cluster 3
    plot_mod = mean_bias[0].plot.pcolormesh(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='RdBu')
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
    std_dev.plot(cmap='Reds',vmin=v_min,vmax=v_max)
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')

#ZONAVG
#plot medie zonali
def plot_zonavg(n_rows,n_cols,fig_size,name_models_to_plot,name_dict,dataset_seas_mean,v_min,v_max,title_plot,title_pdf):
    #plot medie annuali dei modelli
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=fig_size)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Aggiungo spazi verticali tra le subplots
    # Plot dei modelli
    for i in range(n_rows): #ciclo sulle righe
        for j in range(n_cols): #ciclo sulle colonne
            models_index_list = i * n_cols + j #indice del modello all'interno della lista
            if models_index_list == len(name_models_to_plot):
                break
            model_name = name_models_to_plot[models_index_list]
            data_array = name_dict[model_name]['zonavg bias DJF']
            plot_mod = data_array.plot(ax=ax[i, j])
            # Fisso la scala
            plot_mod.set_clim(vmin=v_min, vmax=v_max)
            #Plot della climatologia dei singoli mdoelli e di ERA5
            data = name_dict[model_name]['zonavg seasonal mean DJF']     
            data_era = dataset_seas_mean[4]
            #plot
            data.sel(lon=0).plot.contour(ax=ax[i,j],colors='k') #in modo t.c l'array sia 2d su plev e lat
            data_era.sel(lon=0).plot.contour(ax=ax[i,j],colors='g')
            ax[i,j].set_ylabel('plev')
            ax[i,j].set_xlabel('lat')
            ax[i,j].set_title(model_name) #nome di ogni singolo modello sul plot
            # Inverto l'asse y in modo t.c i livelli di pressione siano corretti
            ax[i,j].invert_yaxis()

    #Rimuovo i quadrati non utilizzati
    for i in range(n_rows):
        for j in range(n_cols):
            models_index_list = i * n_cols + j
            if models_index_list >= len(model_name):
                ax[i, j].axis('off')
    
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)

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
        plot_mod = data_array.plot(ax=ax[i])
        # Fisso la scala
        plot_mod.set_clim(vmin=v_min, vmax=v_max)        
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
    
    # Legenda per linee tratteggiate
    fig.legend(['Linee nere - climatologia modello', 'Linee verdi - climatologia ERA5'], loc='upper right', bbox_to_anchor=(1.2, 1))
    # Titolo
    fig.suptitle(title_plot, fontsize=16, y=1.02)
    fig.savefig(title_pdf, format='pdf')

#plot dei cluster medi per medie zonali

#funzione per il plot dei cluster medi tos
def plot_mean_cluster_zonavg(number_models,name_models_to_plot,name_dict,title_plot,title_pdf,v_min,v_max,fig_size):
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
    mean_zonavg.plot(figsize=fig_size,vmin=v_min, vmax=v_max,cmap='RdBu')
    plt.ylabel('latitude')
    plt.xlabel('longitude')
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
    plt.gca().invert_yaxis()
    # Titolo
    plt.suptitle(title_plot, fontsize=16, y=1.02)

    plt.savefig(title_pdf, format='pdf')