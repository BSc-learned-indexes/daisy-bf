python3 bloom_filters/daisy_BF.py --data_path experiments/url/rfc_good/qx_inv_px/1_per_mille_keys/exported_urls_subsample_345_pos.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True    


python3 bloom_filters/Ada-BF.py --data_path ./experiments/url/rfc_good/qx_inv_px/1_per_mille_keys/exported_urls_subsample_345_pos.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 521 --max_size 3098 --step 500 --Q_dist True

python3 ./bloom_filters/PLBF.py --data_path ./experiments/url/rfc_good/qx_inv_px/1_per_mille_keys/exported_urls_subsample_345_pos.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 521 --max_size 3098 --step 500 --Q_dist True

python3 bloom_filters/Bloom_filter.py --data_path ./experiments/url/rfc_good/qx_inv_px/1_per_mille_keys/exported_urls_subsample_345_pos.csv --min_size 521 --max_size 3098 --step 500 --Q_dist True
