#!/bin/bash

maindir=/archive/paolo/cmip6/CMIP6/model-output

models=$(ls -d $maindir/*/*)

#echo $models

modeloutput=/work/users/guest/sbeltrami

echo -n > name_model.txt #per assicurarmi che il file model.txt sia vuoto all'inizio
echo -n > tos_nc_model.txt
echo -n > remap_tos_nc_model.txt

for modelpath in $models ; do

        model=$(basename $modelpath) #nome del modello

        #echo $(ls $model)

        #echo $model > model.txt
        printf "%s\n" $model >> name_model.txt #stampo il nome del modello su file

        modelfiles=$modelpath/historical/ocean/Omon/r1i1p1f1/tos/tos_Omon_${model}_historical_r1i1p1f1_*.nc
        
        #echo $(ls $modelfiles)

        cdo -cat $modelfiles ${modeloutput}/${model}.nc

        printf "%s\n" ${modeloutput}/${model}.nc >> tos_nc_model.txt #scrivo su file il nome .nc

        cdo remapbil,r180x90 -selmate,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remap.nc
        if [ $? -eq 0 ]; then
            remapcon,r180x90 -selmate,tos ${modeloutput}/${model}.nc ${modeloutput}/${model}_remap.nc
        fi


        printf "%s\n" ${modeloutput}/${model}_remap.nc >> remap_tos_nc_model.txt #scrivo su file il nome .nc

done