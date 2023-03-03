all:
	@python3 url_vectorizor.py
	@python3 create_model.py
	@python3 plot_distributions.py

test:
	@python3 create_model.py --rfc_max_dept 2 --rfc_n_estimators 17
	@python3 plot_distributions.py
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 4 --num_group_max 8 --mem_min 150000 --mem_max 2000000 --mem_step 100000
