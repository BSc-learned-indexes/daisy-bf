vectorize:
	@python3 url_vectorizor.py

zipf:
	@python3 create_synthetic_dataset.py
	@python3 plot_distributions.py --data_set_name syntetic_zipfean

model:
	@python3 create_model.py --model_type random_forest
	@python3 plot_distributions.py --data_set_name exported_urls

qx:
	@python3 decorate_with_qx.py --file_name exported_urls

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True

tau: 
	@python3 bloom_filters/daisy_BF_tau.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 13 --normalize True

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

plbf:
	@python3 ./bloom_filters/PLBF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True

plot_all:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv Standard_BF.csv daisy-BF_tau.csv

plot_learned_bf:
	@python3 plot_size_FPR.py --file_names PLBF_mem_FPR.csv daisy-BF.csv Ada-BF.csv daisy-BF_tau.csv

heatmaps:
	@python3 create_kx_heatmaps.py --file_name PLBF_regions_positives
	@python3 create_kx_heatmaps.py --file_name PLBF_regions_negatives
	@python3 create_kx_heatmaps.py --file_name ADA-BF_regions_positives
	@python3 create_kx_heatmaps.py --file_name ADA-BF_regions_negatives
	@python3 create_kx_heatmaps.py --file_name daisy-BF_k_insert --is_daisy True
	@python3 create_kx_heatmaps.py --file_name daisy-BF_k_lookup --is_daisy True
	@python3 create_kx_heatmaps.py --file_name daisy-BF_tau_k_insert --is_daisy True
	@python3 create_kx_heatmaps.py --file_name daisy-BF_tau_k_lookup --is_daisy True

all:
	@python3 bloom_filters/Bloom_filter.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --min_size 12500 --max_size 26000 --step 1000 --Q_dist True
	@python3 bloom_filters/daisy_BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True
	@python3 bloom_filters/daisy_BF_tau.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 13 --normalize True
	@python3 bloom_filters/Ada-BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True
	@python3 ./bloom_filters/PLBF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 12500 --max_size 26000 --step 1000 --Q_dist True


