#!/bin/bash

maindir=/archive/paolo/cmip6/CMIP6/model-output

models=$(ls -d $maindir/*/*)

#echo $models

modeloutput=/work/users/guest/sbeltrami/atmos

#per assicurarmi che i file siano vuoti all'inizio
echo -n > atmos_name_model.txt #file in cui scrivo il nome di tutti i modelli
echo -n > name_atmos_model.txt #file in cui scrivo il nome dei modelli che hanno modulo atmosfera
echo -n > path_atmos_model.txt #file in cui scrivo il percorso dei modelli che hanno modulo atmosfera
echo -n > path_remap_atmos_model.txt #file in cui scrivo il percorso dei modelli che sono stati rimappati
echo -n > atmos_remapcon_model.txt #file in cui scrivo il nome dei modelli che hanno modulo atmosfera e sono stati rimappati con remapcon
echo -n > atmos_remapbil_model.txt #file in cui scrivo il nome dei modelli che hanno modulo atmosfera e sono stati rimappati con remapbil
echo -n > atmos_remapnn_model.txt #file in cui scrivo il nome dei modelli che hanno modulo atmosfera e sono stati rimappati con remapnn
echo -n > name_no_atmos_model.txt #file in cui scrivo il nome dei modelli che non hanno modulo atmosfera

for modelpath in $models ; do

        model=$(basename $modelpath) #nome del modello

        #echo $(ls $model)
        #Non prendo 5 modelli
        if [ "$model" == "CIESM" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "EC-Earth3-Veg-LR" ] || [ "$model" == "KACE-1-0-G" ] || [ "$model" == "E3SM-1-1" ] || [ "$model" == "E3SM-1-1-ECA" ]; then # CIESM per Value Error index must be monotonic increasing or decreasing, FGOALS per stesso valore medio di temperatura pari a 3.1070e+34
            echo 'CIESM, FGOALS-f3-L, EC-Earth3-Veg-LR, Kace-1-0-G, E3SM-1-1, E3SM-1-1-ECA models not taken'
        
        else
            printf "%s\n" $model >> atmos_name_model.txt #stampo il nome del modello su file
                               
            modelfiles=$modelpath/historical/atmos/Amon/r1i1p1f1/ua/ua_Amon_${model}_historical_r1i1p1f1_*.nc #considero la componente zonale del vento ua

            #modelli
            if ls $modelfiles 1> /dev/null 2>&1; then #prendo più file .nc nel percorso, non soltanto uno

                printf "%s\n" $model >> name_atmos_model.txt #stampo su file il nome del modello con modulo atmosfera 

                cdo -sellevel,85000 -cat $modelfiles ${modeloutput}/${model}.nc #concatena

                printf "%s\n" ${modeloutput}/${model}.nc >> path_atmos_model.txt #scrivo su file il nome .nc

                #remapcon
                cdo remapcon,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapcon.nc           
                if [ $? -eq 0 ]; then #se il comando è stato eseguito correttamente allora scrivo su file
                    printf "%s\n" ${modeloutput}/${model}_remapcon.nc >> path_remap_atmos_model.txt
                    printf "%s\n" $model >> atmos_remapcon_model.txt #scrivo su file il nome .nc
                else #se il comando precedente non è stato eseguito con successo
                    rm ${modeloutput}/${model}_remapcon.nc #rimuovo quanto creato perché non è andato a buon fine il processo
                    #remapbil
                    cdo remapbil,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapbil.nc
                    if [ $? -eq 0 ]; then 
                        printf "%s\n" ${modeloutput}/${model}_remapbil.nc  >> path_remap_atmos_model.txt
                        printf "%s\n" $model >> atmos_remapbil_model.txt #scrivo su file il nome .nc
                    else #se il comando precedente non è stato eseguito con successo
                        rm ${modeloutput}/${model}_remapbil.nc
                        #remapnn
                        cdo remapnn,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapnn.nc
                        if [ $? -eq 0 ]; then  
                            printf "%s\n" ${modeloutput}/${model}_remapnn.nc  >> path_remap_atmos_model.txt
                            printf "%s\n" $model >> atmos_remapnn_model.txt #scrivo su file il nome .nc
                        else
                            rm ${modeloutput}/${model}_remapnn.nc
                            echo "${model} non rimappabile con i tre remap scelti"
                        fi
                    fi                
                fi


            else

                printf "%s\n" $model >> name_no_atmos_model.txt #modelli che non contengono il modulo atmosfera
            fi
        fi


done