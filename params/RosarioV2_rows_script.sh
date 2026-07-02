#! /bin/bash

# RUN THIS SCRIPT FROM THE MAIN WORKSPACE FOLDER
PARAMS_ROW2="../params/aligner_RosarioV2_params_row2.yaml"
PARAMS_ROW3="../params/aligner_RosarioV2_params_row3.yaml"
# 10 100 50 == 0.1 scale noise, 250 mm translational noise, 5º Y noise
cd ../bin/
# for TranslNoise in $(seq 0 100 200):
# do
	# for YNoise in $(seq 0):
	# do
		# for Scale in $(seq 00 10 50):
		# do
	
		# 	for ExpID in $(seq 0 1 5):
		# 	do	
		# 		echo ${PARAMS_ROW2} ${Scale} 100 0 ${ExpID} 11
		# 		./registration_node ${PARAMS_ROW2} ${Scale} 100 0 ${ExpID} 11
		# 		echo ${PARAMS_ROW3} ${Scale} 100 0 ${ExpID} 11
		# 		./registration_node ${PARAMS_ROW3} ${Scale} 100 0 ${ExpID} 11
		# 	done	
		# done
	# done
# done


# for TranslNoise in $(seq 0 100 200):
# do	
	# for YNoise in 0 10 20:
	# do
		for Scale in $(seq 20 10 40):
		do
				 for Mode in 0 4 7 8 12 13 14 :
			 do
				for ExpID in $(seq 0 1 5):
				do	
					echo ${PARAMS_ROW2} ${Scale} 100 10 ${ExpID} ${Mode}
					./registration_node ${PARAMS_ROW2} ${Scale} 100 10 ${ExpID} ${Mode}					
				done
			 done
		done
	# done
# done

# for Scale in $(seq 0 5 30):
# do

# 	# for YNoise in $(seq 0):
# 	# do
# 		for ExpID in $(seq 0 1 2):
# 		do	
# 			echo ${PARAMS_ROW3} ${Scale} 100 10 ${ExpID} 0
# 			./registration_node ${PARAMS_ROW3} ${Scale} 100 10 ${ExpID} 0
# 		done
# 	# done
# done

# for Scale in $(seq 0 10 10):
# do
# 	for TranslNoise in $(seq 80 10 170):
# 	do
# 		for YNoise in $(seq 30 5 50):
# 		do
# 			for ExpID in $(seq 0 1 1):
# 			do	
# 				echo ${PARAMS_ROW3} 0 ${TranslNoise} ${YNoise} ${ExpID} 0
# 				./registration_node ${PARAMS_ROW3} 0 ${TranslNoise} ${YNoise} ${ExpID} 0
# 			done
# 		done
# 	done
# done