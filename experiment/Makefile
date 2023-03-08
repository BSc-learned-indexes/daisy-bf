model:
	@python3 url_vectorizor.py
	@python3 create_model.py
	@python3 plot_distributions.py

PLBF:
	@python3 create_model.py --rfc_max_dept 2 --rfc_n_estimators 17
	@python3 plot_distributions.py
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 3 --num_group_max 20 --min_size 100000 --max_size 500000 --step 50000


standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/exported_urls.csv

Ada-BF:
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/exported_urls.csv --num_group_min 3 --num_group_max 20 --min_size 100000 --max_size 500000 --step 50000


plot_all:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv Ada-BF.csv
