
python3 bloom_filters/daisy_BF.py --data_path experiments/zipfean/qx_eq_px/syntetic_zipfean.csv --fpr_data_path ./data/plots/Ada-BF.csv --model_path ./models/model.pickle --tau False --max_iter 30 --normalize True

python3 bloom_filters/Bloom_filter.py --data_path experiments/zipfean/qx_eq_px/syntetic_zipfean.csv --min_size 2680 --max_size 29105 --step 1000

python3 ./bloom_filters/PLBF.py --data_path experiments/zipfean/qx_eq_px/syntetic_zipfean.csv --model_path ./models/model.pickle --num_group_min 2 --num_group_max 8 --min_size 2680 --max_size 29105 --step 3000 


python3 bloom_filters/Ada-BF.py --data_path experiments/zipfean/qx_eq_px/syntetic_zipfean.csv --model_path ./models/model.pickle --num_group_min 8 --num_group_max 12 --c_min 2.0 --c_max 2.5 --min_size 2680 --max_size 29105 --step 3000 