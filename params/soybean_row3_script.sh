#! /bin/bash
# RUN THIS SCRIPT FROM THE MAIN WORKSPACE FOLDER
PARAMS_FILE_PATH1="../params/aligner_soybean_params_row3.yaml"
PARAMS_FILE_PATH2="../params/aligner_soybean_params_row4.yaml"
PARAMS_FILE_PATH3="../params/aligner_soybean_params_row5.yaml"

PARAMS_LOG="../params/log.txt"
# 10 100 50 == 0.1 scale noise, 250 mm translational noise, 5º Y noise
cd ../bin/

for TranslNoise in $(seq 0 100 200)
do	
	for YNoise in $(seq 0 10 50)
	do
		for Scale in 10
		do
			 for Mode in 0 1 2 4 8 12 13 14 15 16 17 18 19 29 21 22 23 24
			 do
				for ExpID in $(seq 0 1 15)
				do	
					echo $(date) ${PARAMS_FILE_PATH1} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode} >> ${PARAMS_LOG}
					echo ${PARAMS_FILE_PATH1} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode}					
					./registration_node ${PARAMS_FILE_PATH1} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode}
					
					echo $(date) ${PARAMS_FILE_PATH3} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode} >> ${PARAMS_LOG}
					echo ${PARAMS_FILE_PATH3} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode}
					./registration_node ${PARAMS_FILE_PATH3} ${Scale} ${TranslNoise} ${YNoise} ${ExpID} ${Mode}
				done
			 done
		done
	done	
done
# Probar utilizarlo un solo frame. Y si eso funciona, modificarlo para alinear varios frame  independientemente.  Esto tiene la desventaja de que no tiene en cuenta temporalidad o resultados anteriores.

