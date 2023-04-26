model:
	# @python3 url_vectorizor.py
	# @python3 create_model.py
	@python3 create_model.py --rfc_max_dept 2 --rfc_n_estimators 10
	@python3 plot_distributions.py

plbf:
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/daisy-out.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 1000 --max_size 70000 --step 5000

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/syntetic_zipfean.csv --min_size 1000 --max_size 70000 --step 5000

daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/syntetic_zipfean.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau True --max_iter 30 --normalize True --use_Q_dist True
	#@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau True --max_iter 30 --normalize True

bb_daisy: 
	@python3 bloom_filters/blackboard_thresholds_daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/PLBF_mem_FPR.csv --model_path ./models/model.pickle

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/syntetic_zipfean.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 10000 --max_size 70000 --step 10000

learned: 
	@python3 bloom_filters/learned_Bloom_filter.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --min_size 150000 --max_size 500000 --step 50000

plot_all:
	@python3 plot_size_FPR.py --file_names Ada-BF.csv daisy-BF.csv Standard_BF.csv PLBF_mem_FPR.csv

plot_learned_bf:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv 

build_all_big:
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/exported_urls.csv --max_size 700000
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 4 --num_group_max 12 --min_size 150000 --max_size 700000 --step 50000
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 150000 --max_size 700000 --step 50000


