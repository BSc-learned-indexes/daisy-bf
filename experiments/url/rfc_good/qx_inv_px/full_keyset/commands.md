	python3 bloom_filters/Ada-BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 12520 --max_size 25094 --step 1000 --Q_dist True

	python3 bloom_filters/daisy_BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True

	python3 ./bloom_filters/PLBF.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 12520 --max_size 25094 --step 1000 --Q_dist True

	python3 bloom_filters/Bloom_filter.py --data_path ./experiments/url/rfc_good/qx_inv_px/full_keyset/exported_urls_qx_inv_px_full_keyset.csv --min_size 12520 --max_size 25094 --step 1000 --Q_dist True
