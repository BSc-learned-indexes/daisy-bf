plbf:
	@python3 ./bloom_filters/PLBF.py --data_path data/scores/exported_urls_subsample_3457_pos.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 7 --min_size 9000 --max_size 24000 --step 1500 --Q_dist True

standard: 
	@python3 bloom_filters/Bloom_filter.py --data_path data/scores/exported_urls_subsample_3457_pos.csv --min_size 9000 --max_size 24000 --step 1500 --Q_dist True

daisy: 
	@python3 bloom_filters/daisy_BF.py --data_path data/scores/exported_urls_subsample_3457_pos.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True --Q_dist True
	#@python3 bloom_filters/daisy_BF.py --data_path ./data/scores/exported_urls.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau True --max_iter 30 --normalize True

adabf:
	@python3 bloom_filters/Ada-BF.py --data_path data/scores/exported_urls_subsample_3457_pos.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 1.6 --c_max 2.5 --min_size 9000 --max_size 24000 --step 1500 --Q_dist True
