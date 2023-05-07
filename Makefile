vectorize:
	@python3 url_vectorizor.py

zipf:
	@python3 create_synthetic_dataset.py
	@python3 plot_distributions.py --data_set_name syntetic_zipfean

model:
	@python3 create_model.py --model_type regression
	# @python3 create_model.py --rfc_n_estimators 10
	@python3 plot_distributions.py --data_set_name exported_urls

qx:
	@python3 decorate_with_qx.py --file_name exported_urls

plbf:
	@python3 ./bloom_filters/PLBF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True
	#@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau True --max_iter 30 --normalize True

tau: 
	@python3 bloom_filters/daisy_BF_tau.py --data_path ./data/scores/exported_urls_subsample_3457_pos.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 13 --normalize True

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

adabf2:
	@python3 bloom_filters/Ada-BF2.py --data_path ./data/scores/syntetic_zipfean.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 8 --c_min 2.0 --c_max 2.0 --min_size 10000 --max_size 10000 --step 10000

learned: 
	@python3 bloom_filters/learned_Bloom_filter.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --min_size 150000 --max_size 500000 --step 50000

plot_all:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv Standard_BF.csv daisy-BF_tau.csv

plot_learned_bf:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv daisy-BF_tau.csv
	#@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv

heatmaps:
	@python3 create_kx_heatmaps.py --file_name daisy-BF_tau_k_insert --is_daisy True
	@python3 create_kx_heatmaps.py --file_name daisy-BF_tau_k_lookup --is_daisy True
	@python3 create_kx_heatmaps.py --file_name PLBF_regions_positives
	@python3 create_kx_heatmaps.py --file_name PLBF_regions_negatives
	@python3 create_kx_heatmaps.py --file_name ADA-BF_regions_positives
	@python3 create_kx_heatmaps.py --file_name ADA-BF_regions_negatives
	@python3 create_kx_heatmaps.py --file_name daisy-BF_k_insert --is_daisy True
	@python3 create_kx_heatmaps.py --file_name daisy-BF_k_lookup --is_daisy True

build_all_big:
	@python3 bloom_filters/Bloom_filter.py --data_path ./data/scores/exported_urls.csv --max_size 700000
	@python3 ./bloom_filters/PLBF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 4 --num_group_max 12 --min_size 150000 --max_size 700000 --step 50000
	@python3 bloom_filters/Ada-BF.py --data_path ./data/scores/exported_urls.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 150000 --max_size 700000 --step 50000


