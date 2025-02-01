dataset=imdb
lime_num_samples=1000
experiment_dir=experiments/${dataset}/lime/k_all-n_${lime_num_samples}-mask_remove

sampler=uniform
num_samples=500
aggregator=norm_lime

bash run.sh scripts/visualize_selected_explanations.py \
    --explanation_file $experiment_dir/explanations.h5 \
    --dataset $dataset \
    --num_samples $num_samples \
    --sampler $sampler \
    --aggregator  $aggregator \
    --output_file $experiment_dir/sampler_${sampler}-n_${num_samples}-aggregator_${aggregator}

