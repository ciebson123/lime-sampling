dataset=yelp
num_samples=1000

for num_features in 10 all; do
    if [ $num_features = "all" ]; then
        num_features_arg=""
    else
        num_features_arg="--lime_num_features $num_features"
    fi

    for token_masking_strategy in remove; do
        sbatch run.sh scripts/calculate_local_explanations.py \
            --dataset $dataset \
            --experiment_dir "experiments/${dataset}/lime/k_${num_features}-n_${num_samples}-mask_${token_masking_strategy}" \
            --batch_size 1000 \
            --lime_num_samples $num_samples \
            --lime_token_masking_strategy $token_masking_strategy \
            $num_features_arg
    done
done
