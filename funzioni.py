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
    annual_mean_weighted = dataset_weighted.mean(dim=("lon", "lat")).resample(time='Y').mean(dim='time')
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
    dataset_resample = dataset.sel(time=slice(year_start,year_end)).resample(time='Q-MAR').mean(dim='time')
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


#funzione che calcola il bias