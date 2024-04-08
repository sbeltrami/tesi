#!/bin/bash

maindir=/archive/paolo/cmip6/CMIP6/model-output

models=$(ls -d $maindir/*/*)

#echo $models

modeloutput=/work/users/guest/sbeltrami/zonal_mean

#per assicurarmi che i file siano vuoti all'inizio
echo -n > zonal_mean_name_model.txt #tutti i modelli
echo -n > zonal_mean_name_ok_model.txt # modelli che hanno il percorso indicato
echo -n > zonal_mean_name_no_model.txt #modelli che non hanno il percorso indicato
echo -n > path_zonal_mean_model.txt #percorso dei file
echo -n > path_remap_zonal_mean_model.txt #nome dei file con remap 
echo -n > zonal_mean_remapcon_model.txt #nome modelli con remapcon
echo -n > zonal_mean_remapbil_model.txt #nome modelli con remapbil
echo -n > zonal_mean_remapnn_model.txt #nome modelli con remapnn

for modelpath in $models ; do

        model=$(basename $modelpath) #nome del modello

        #echo $(ls $model)
        #Non prendo 5 modelli
        if [ "$model" == "CIESM" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "EC-Earth3-Veg-LR" ] || [ "$model" == "KACE-1-0-G" ] || [ "$model" == "E3SM-1-1" ] || [ "$model" == "E3SM-1-1-ECA" ] || [ "$model" == "NorCPM1" ]; then # CIESM per Value Error index must be monotonic increasing or decreasing, FGOALS per stesso valore medio di temperatura pari a 3.1070e+34
            echo 'CIESM, FGOALS-f3-L, EC-Earth3-Veg-LR, Kace-1-0-G, E3SM-1-1, E3SM-1-1-ECA, NorCMP1 models not taken'
        
        else
            printf "%s\n" $model >> zonal_mean_name_model.txt #stampo il nome del modello su file
                               
            modelfiles=$modelpath/historical/atmos/Amon/r1i1p1f1/ua/ua_Amon_${model}_historical_r1i1p1f1_*.nc #considero la componente zonale del vento ua

            #modelli
            if ls $modelfiles 1> /dev/null 2>&1; then #prendo più file .nc nel percorso, non soltanto uno

                printf "%s\n" $model >> zonal_mean_name_ok_model.txt #stampo su file il nome del modello con modulo atmosfera 

                cdo -cat $modelfiles ${modeloutput}/${model}.nc #concatena --> devo selezionare emisfero nord + fare la media zonale

                printf "%s\n" ${modeloutput}/${model}.nc >> path_zonal_mean_model.txt #scrivo su file il nome .nc

                #remapcon
                cdo remapcon,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapcon.nc           
                if [ $? -eq 0 ]; then #se il comando è stato eseguito correttamente allora scrivo su file                    
                    cdo zonavg -sellonlatbox,-60,0,0,90 ${modeloutput}/${model}_remapcon.nc ${modeloutput}/${model}_zonal_mean.nc #calcolo la media zonale dell'emisfero nord tra -60W e 0 
                    printf "%s\n" ${modeloutput}/${model}_remapcon.nc >> path_remap_zonal_mean_model.txt
                    printf "%s\n" $model >> zonal_mean_remapcon_model.txt #scrivo su file il nome .nc
                else #se il comando precedente non è stato eseguito con successo
                    rm ${modeloutput}/${model}_remapcon.nc #rimuovo quanto creato perché non è andato a buon fine il processo
                    #remapbil
                    cdo remapbil,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapbil.nc
                    if [ $? -eq 0 ]; then 
                        cdo fldmean -sellonlatbox,-60,0,0,90 ${modeloutput}/${model}_remapbil.nc ${modeloutput}/${model}_zonal_mean.nc #calcolo la media zonale dell'emisfero nord tra -60W e 0 
                        printf "%s\n" ${modeloutput}/${model}_remapbil.nc  >> path_remap_zonal_mean_model.txt
                        printf "%s\n" $model >> zonal_mean_remapbil_model.txt #scrivo su file il nome .nc
                    else #se il comando precedente non è stato eseguito con successo
                        rm ${modeloutput}/${model}_remapbil.nc
                        #remapnn
                        cdo remapnn,r180x90 -selname,ua ${modeloutput}/${model}.nc ${modeloutput}/${model}_remapnn.nc
                        if [ $? -eq 0 ]; then  
                            cdo fldmean -sellonlatbox,-60,0,0,90 ${modeloutput}/${model}_remapnn.nc ${modeloutput}/${model}_zonal_mean.nc #calcolo la media zonale dell'emisfero nord tra -60W e 0 
                            printf "%s\n" ${modeloutput}/${model}_remapnn.nc  >> path_remap_zonal_mean_model.txt
                            printf "%s\n" $model >> zonal_mean_remapnn_model.txt #scrivo su file il nome .nc
                        else
                            rm ${modeloutput}/${model}_remapnn.nc
                            echo "${model} non rimappabile con i tre remap scelti"
                        fi
                    fi                
                fi
                
            else

                printf "%s\n" $model >> zonal_mean_name_no_model.txt #modelli che non contengono il modulo atmosfera
            fi
        fi


done