#!/bin/bash

maindir=/archive/paolo/cmip6/CMIP6/model-output

models=$(ls -d $maindir/*/*)

#echo $models

modeloutput=/work/users/guest/sbeltrami

#per assicurarmi che i file siano vuoti all'inizio
echo -n > name_model.txt #file in cui scrivo il nome di tutti i modelli
echo -n > tos_nc_model.txt #file in cui scrivo il percorso dei modelli che hanno modulo oceano
echo -n > remap_tos_nc_model.txt #file in cui scrivo il percorso dei modelli che hanno modulo oceano e sono stati rimappati
echo -n > name_model_ocean.txt #file in cui scrivo il nome dei modelli che hanno modulo oceano
echo -n > no_ocean.txt #file in cui scrivo il nome dei modelli che non hanno modulo oceano

for modelpath in $models ; do

        model=$(basename $modelpath) #nome del modello

        #echo $(ls $model)

        printf "%s\n" $model >> name_model.txt #stampo il nome del modello su file

        modelfiles=$modelpath/historical/ocean/Omon/r1i1p1f1/tos/tos_Omon_${model}_historical_r1i1p1f1_*.nc
        
        #echo $(ls $modelfiles)
        if [ -e $modelfiles ]; then #se il percorso esiste

            printf "%s\n" $model >> name_model_ocean.txt #stampo su file il nome del modello con oceano 

            cdo -cat $modelfiles ${modeloutput}/${model}.nc #concatena

            printf "%s\n" ${modeloutput}/${model}.nc >> tos_nc_model.txt #scrivo su file il nome .nc

            cdo remapbil,r180x90 -selname,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remap.nc
            if [ $? -eq 0 ]; then #se il comando precedente non è stato eseguito con successo
                cdo remapcon,r180x90 -selname,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remap.nc
            fi

            printf "%s\n" ${modeloutput}/${model}_remap.nc >> remap_tos_nc_model.txt #scrivo su file il nome .nc

        else

            printf "%s\n" "$model non contiene il modulo ocean" >> no_ocean.txt
        fi


done