model:
	# @python3 url_vectorizor.py
	@python3 create_model.py --rfc_max_dept 4 --rfc_n_estimators 6
	@python3 plot_distributions.py

pblf:
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 3 --num_group_max 20 --min_size 150000 --max_size 1500000 --step 50000

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/exported_urls.csv

daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/PLBF_mem_FPR.csv --model_path ./models/model.pickle

bb_daisy: 
	@python3 bloom_filters/blackboard_thresholds_daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/PLBF_mem_FPR.csv --model_path ./models/model.pickle

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 150000 --max_size 500000 --step 50000

learned: 
	@python3 bloom_filters/learned_Bloom_filter.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --min_size 150000 --max_size 500000 --step 50000

plot_all:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv Ada-BF.csv 

plot_learned_bf:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv Ada-BF.csv daisy-BF.csv

build_all_big:
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/exported_urls.csv --max_size 1500000
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 3 --num_group_max 20 --min_size 150000 --max_size 1500000 --step 50000
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 150000 --max_size 1500000 --step 50000


