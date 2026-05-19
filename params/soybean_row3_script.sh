#! /bin/bash

# RUN THIS SCRIPT FROM THE MAIN WORKSPACE FOLDER
PARAMS_FILE_PATH="../params/aligner_soybean_params_row5.yaml"
# 10 100 50
cd ../bin/
for TranslNoise in $(seq 0 100 200):
do
	for Scale in $(seq 0 10 30):
	do
	
		# for YNoise in $(seq 0):
		# do
			for ExpID in $(seq 0 1 2):
			do	
				echo ${PARAMS_FILE_PATH} ${Scale} ${TranslNoise} 0 ${ExpID} 0
				./registration_node ${PARAMS_FILE_PATH} ${Scale} ${TranslNoise} 0 ${ExpID} 0
			done
		# done
	done
done

