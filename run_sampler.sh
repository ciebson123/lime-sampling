export PYTHONPATH="${PYTHONPATH}:$(pwd)"

for i in 10 20 50 100 200 500 1000; 
    do 
    python scripts/calculate_global_explanation.py --explanation_file storage/emotion_lime/lime/k_all-n_5000-mask_remove/explanations.h5 --num_samples $i --output_file storage/ent_exp/entropy_explanations_$i.json --sampler entropy ; 
    echo "$i done";
    done