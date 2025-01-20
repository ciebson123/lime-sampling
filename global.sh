dataset=emotion
lime_num_samples=5000
source .env
echo proj_dir=${PROJECT_DIR}
experiment_dir=${PROJECT_DIR}/experiments/${dataset}/lime/k_all-n_${lime_num_samples}-mask_remove

for aggregator in norm_lime; do
    # ground truth
    bash run.sh scripts/calculate_global_explanation.py \
        --explanation_file $experiment_dir/explanations.h5 \
        --dataset $dataset \
        --num_samples -1 \
        --aggregator  $aggregator \
        --output_file $experiment_dir/global_explanation_full-aggregator_${aggregator}.json

    # sampled
    for num_samples in 7 15 31 62 125 250 500 1000; do 
    # for num_samples in 10 20 50 100 200 500 1000 2000 5000 10000; do 
        for sampler in kernel_thinning uniform entropy el2n variation_ratio; do
            echo "Running sampler $sampler with $num_samples samples"
            bash run.sh scripts/calculate_global_explanation.py \
                --explanation_file $experiment_dir/explanations.h5 \
                --dataset $dataset \
                --num_samples $num_samples \
                --sampler $sampler \
                --aggregator  $aggregator \
                --output_file $experiment_dir/global_explanation-sampler_${sampler}-n_${num_samples}-aggregator_${aggregator}.json
        done
    done    
done
