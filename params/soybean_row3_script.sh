#! /bin/bash

# RUN THIS SCRIPT FROM THE MAIN WORKSPACE FOLDER
PARAMS_FILE_PATH1="../params/aligner_soybean_params_row3.yaml"
PARAMS_FILE_PATH2="../params/aligner_soybean_params_row4.yaml"
PARAMS_FILE_PATH3="../params/aligner_soybean_params_row5.yaml"
# 10 100 50 == 0.1 scale noise, 250 mm translational noise, 5º Y noise
cd ../bin/
# for TranslNoise in $(seq 0 100 200):
# do
# 	for Scale in $(seq 0 5 30):
# 	do
	
# 		# for YNoise in $(seq 0):
# 		# do
# 			for ExpID in $(seq 0 1 2):
# 			do	
# 				echo ${PARAMS_FILE_PATH1} ${Scale} 100 10 ${ExpID} 0
# 				./registration_node ${PARAMS_FILE_PATH1} ${Scale} 100 10 ${ExpID} 0
# 			done
# 		# done
# 	done
# # done

# for Scale in $(seq 0 5 30):
# do

# 	# for YNoise in $(seq 0):
# 	# do
# 		for ExpID in $(seq 0 1 2):
# 		do	
# 			echo ${PARAMS_FILE_PATH2} ${Scale} 100 10 ${ExpID} 0
# 			./registration_node ${PARAMS_FILE_PATH2} ${Scale} 100 10 ${ExpID} 0
# 		done
# 	# done
# done

# for Scale in $(seq 0 10 10):
# do
	for TranslNoise in $(seq 80 10 170):
	do
		for YNoise in $(seq 30 5 50):
		do
			for ExpID in $(seq 0 1 1):
			do	
				echo ${PARAMS_FILE_PATH3} 0 ${TranslNoise} ${YNoise} ${ExpID} 0
				./registration_node ${PARAMS_FILE_PATH3} 0 ${TranslNoise} ${YNoise} ${ExpID} 0
			done
		done
	done
# done