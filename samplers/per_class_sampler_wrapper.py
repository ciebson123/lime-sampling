import h5py
import tempfile

def wrap_sampler(sampler):
    def wrapper(file: h5py.File, num_samples: int, *args, **kwargs) -> list:
        total_rows = len(file)
        class_to_rows = {}
        for key, value in file.items():
            class_idx = value["predicted_label_idx"][()]
            if class_idx not in class_to_rows:
                class_to_rows[class_idx] = []
            class_to_rows[class_idx].append(key)
        samples = []
        for class_idx in class_to_rows:
            with tempfile.TemporaryFile() as temp_file:
                with h5py.File(temp_file, "w") as subfile:
                    for row in class_to_rows[class_idx]:
                        file.copy(row, subfile)
                    num_subsamples = round(
                        num_samples * len(class_to_rows[class_idx]) / total_rows
                    )
                    num_subsamples = max(num_subsamples, 1)
                    num_subsamples = min(num_subsamples, len(class_to_rows[class_idx]))
                    samples.extend(sampler(subfile, num_subsamples, *args, **kwargs))
        return samples
    return wrapper
