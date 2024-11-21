# TODO: using selected aggregation method
# TODO: calculate ground truth (all samples) global explanation
# TODO: calculate compressed global explanation for some sample selection method
# TODO: compare both global explanations (MAE or TV)


import h5py

with h5py.File("explanations.h5", "r") as f:
	for sample_idx, grp in f.items():
		print(grp.keys())
		
		print(type(grp["label_idx"][()]))
		print(type(grp["predicted_label_idx"][()]))
		print(type(grp["explanation_fit"][()]))
		print(type(grp["probabilities"][:]))
		print(type(grp["token_ids"][:]))
		print(type(grp["token_scores"][:]))
		break