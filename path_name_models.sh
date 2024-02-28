#!/bin/bash

maindir=/archive/paolo/cmip6/CMIP6/model-output

models=$(ls -d $maindir/*/*)

#echo $models

modeloutput=/work/users/guest/sbeltrami

#per assicurarmi che i file siano vuoti all'inizio
echo -n > name_model.txt #file in cui scrivo il nome di tutti i modelli
echo -n > name_ocean_model.txt #file in cui scrivo il nome dei modelli che hanno modulo oceano
echo -n > path_ocean_model.txt #file in cui scrivo il percorso dei modelli che hanno modulo oceano
echo -n > path_remap_ocean_model.txt #file in cui scrivo il percorso dei modelli che sono stati rimappati
echo -n > remapcon_model.txt #file in cui scrivo il nome dei modelli che hanno modulo oceano e sono stati rimappati con remapcon
echo -n > remapbil_model.txt #file in cui scrivo il nome dei modelli che hanno modulo oceano e sono stati rimappati con remapbil
echo -n > remapnn_model.txt #file in cui scrivo il nome dei modelli che hanno modulo oceano e sono stati rimappati con remapnn
echo -n > name_no_ocean_model.txt #file in cui scrivo il nome dei modelli che non hanno modulo oceano

for modelpath in $models ; do

        model=$(basename $modelpath) #nome del modello

        #echo $(ls $model)
        #Non prendo il modello CIESM
        if [ "$model" == "CIESM" ]; then
            echo 'CIESM model not taken'
        
        else
            printf "%s\n" $model >> name_model.txt #stampo il nome del modello su file

            modelfiles=$modelpath/historical/ocean/Omon/r1i1p1f1/tos/tos_Omon_${model}_historical_r1i1p1f1_*.nc
        
            #echo $(ls $modelfiles)
            #if [ -e $modelfiles ]; then #se il percorso esiste
            if ls $modelfiles 1> /dev/null 2>&1; then #prendo più file .nc nel percorso, non soltanto uno

                printf "%s\n" $model >> name_ocean_model.txt #stampo su file il nome del modello con modulo oceano 

                cdo -cat $modelfiles ${modeloutput}/${model}.nc #concatena

             printf "%s\n" ${modeloutput}/${model}.nc >> path_ocean_model.txt #scrivo su file il nome .nc

                #remapcon
                cdo remapcon,r180x90 -selname,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapcon.nc           
                if [ $? -eq 0 ]; then #se il comando è stato eseguito correttamente allora scrivo su file
                    printf "%s\n" ${modeloutput}/${model}_remapcon.nc >> path_remap_ocean_model.txt
                    printf "%s\n" $model >> remapcon_model.txt #scrivo su file il nome .nc
                else #se il comando precedente non è stato eseguito con successo
                    rm ${modeloutput}/${model}_remapcon.nc #rimuovo quanto creato perché non è andato a buon fine il processo
                    #remapbil
                    cdo remapbil,r180x90 -selname,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapbil.nc
                    if [ $? -eq 0 ]; then 
                        printf "%s\n" ${modeloutput}/${model}_remapbil.nc  >> path_remap_ocean_model.txt
                        printf "%s\n" $model >> remapbil_model.txt #scrivo su file il nome .nc
                    else #se il comando precedente non è stato eseguito con successo
                        rm ${modeloutput}/${model}_remapbil.nc
                        #remapnn
                        cdo remapnn,r180x90 -selname,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapnn.nc
                        if [ $? -eq 0 ]; then  
                            printf "%s\n" ${modeloutput}/${model}_remapnn.nc  >> path_remap_ocean_model.txt
                            printf "%s\n" $model >> remapnn_model.txt #scrivo su file il nome .nc
                        else
                            rm ${modeloutput}/${model}_remapnn.nc
                            echo "${model} non rimappabile con i tre remap scelti"
                        fi
                    fi                
                fi

            else

                printf "%s\n" $model >> name_no_ocean_model.txt #modelli che non contengono il modulo oceano
            fi
        fi


done