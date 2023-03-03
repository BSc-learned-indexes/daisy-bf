all:
	@python3 url_vectorizor.py
	@python3 create_model.py
	@python3 plot_distributions.py

test:
	@python3 create_model.py --rfc_max_dept 2 --rfc_n_estimators 17
	@python3 plot_distributions.py
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 3 --num_group_max 20 --mem_min 100000 --mem_max 500000 --mem_step 50000
	@python2 plot_size_FPR.py --file_names PLBF_mem_FPR_first.csv PLBF_mem_FPR.csv
