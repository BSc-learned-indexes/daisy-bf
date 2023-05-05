daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path experiments/zipfean/qx_inv_px/good_ratio/syntetic_zipfean_good_qx_inv_px.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True
	#@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau True --max_iter 30 --normalize True

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path experiments/zipfean/qx_inv_px/good_ratio/syntetic_zipfean_good_qx_inv_px.csv --min_size 2200 --max_size 29200 --step 2000 --Q_dist True

plbf:
	@python3 ./bloom_filters/PLBF.py --data_path experiments/zipfean/qx_inv_px/good_ratio/syntetic_zipfean_good_qx_inv_px.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 2200 --max_size 29200 --step 2000 --Q_dist True #--min_size 2800 --max_size 17800 --step 1000 --Q_dist True

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path experiments/zipfean/qx_inv_px/good_ratio/syntetic_zipfean_good_qx_inv_px.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 2200 --max_size 29200 --step 2000 --Q_dist True #--min_size 2800 --max_size 17800 --step 1000 --Q_dist True
