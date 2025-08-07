import os
import torch
import torch.multiprocessing as mp # Import multiprocessing

# --- CRITICAL: Set multiprocessing start method IMMEDIATELY ---
# This must happen before any other imports or code that might implicitly
# initialize CUDA context in the main process.
torch.cuda.empty_cache()
try:
    mp.set_start_method('spawn', force=True)
    print(f"Multiprocessing start method set to: {mp.get_start_method()}", flush=True)
except RuntimeError as e:
    print(f"WARNING: Could not set start method to 'spawn': {e}. This might indicate CUDA was already initialized.", flush=True)
    # If it's already set to spawn, that's fine. If it's set to fork and throws, that's a problem.
    if mp.get_start_method(allow_none=True) != 'spawn':
        print("CRITICAL ERROR: 'spawn' method could not be set and current method is not 'spawn'. Expect issues.", flush=True)
        # You might want to exit here, or handle it based on your desired behavior.

# Now, import other libraries after spawn is set
from transformers import pipeline
from datasets import Dataset, load_from_disk, Sequence, Value
from types import SimpleNamespace as Namespace

# --- Configuration ---
config = {
    # No longer specify a single 'cuda_device' for the global pipeline
    # Each process will determine its own device based on rank
    # "path_to_model_on_disk": "/data4/mmendieta/models/ml-e5-large_finetuned_twitter_all_labels/",
    "model_ckpt": "mjwong/multilingual-e5-large-xnli", # The NLI model checkpoint
    "max_length": 32,
    "dataset_name": "/data4/mmendieta/data/geo_corpus.0.0.1_tok_test_ds_e5_inference_results_sample", # Your entire test dataset path
    "batch_size": 1024, # This will be the batch size PER PROCESS
    "num_gpus_to_use": 15, # Specify how many GPUs you want to use
    "fout_nli_csv": "/data4/mmendieta/data/geo_corpus.0.0.1_tok_test_ds_e5_inference_results_nli_sample_multi_gpu.csv" # Output CSV path
}
args = Namespace(**config)

# --- Your provided 40 ground truth labels ---
labels = ['post1geo10', 'post1geo20', 'post1geo30', 'post1geo50', 'post1geo70', 'post2geo10', 'post2geo20',
                  'post2geo30', 'post2geo50', 'post2geo70', 'post3geo10', 'post3geo20', 'post3geo30', 'post3geo50',
                  'post3geo70', 'post7geo10', 'post7geo20', 'post7geo30', 'post7geo50', 'post7geo70', 'pre1geo10',
                  'pre1geo20', 'pre1geo30', 'pre1geo50', 'pre1geo70', 'pre2geo10', 'pre2geo20', 'pre2geo30',
                  'pre2geo50', 'pre2geo70', 'pre3geo10', 'pre3geo20', 'pre3geo30', 'pre3geo50', 'pre3geo70',
                  'pre7geo10', 'pre7geo20', 'pre7geo30', 'pre7geo50', 'pre7geo70']

# --- NLI Hypotheses ---
nli_hypotheses = ["fear", "anger", "joy", "hostility", "hate", "love"]

# --- Function for parallel processing (each worker will call this) ---
def add_nli_scores_to_example_batched_parallel(examples, rank):
    # Each process needs to instantiate its own pipeline
    # And assign it to a unique GPU based on its rank (process ID)
    # This function is run in a *spawned* child process.
    # It must re-initialize everything it needs locally.
    # Check CUDA availability again within the child process.
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        # Ensure we don't request more GPUs than available or specified by user
        gpus_to_use = min(args.num_gpus_to_use, num_available_gpus)
        if gpus_to_use == 0: # Fallback to CPU if no GPUs requested or available
             device = -1
             print(f"Process {os.getpid()} (rank {rank}): No GPUs requested/available. Assigning to CPU.", flush=True)
        else:
            gpu_id = rank % gpus_to_use
            device = f"cuda:{gpu_id}"
            print(f"Process {os.getpid()} (rank {rank}): Initializing NLI pipeline on device: {device}", flush=True)
    else:
        device = -1 # CPU
        print(f"Process {os.getpid()} (rank {rank}): No CUDA available. Initializing NLI pipeline on CPU.", flush=True)

    # Instantiate pipeline locally within each process
    nli_pipe_local = pipeline(
        "zero-shot-classification",
        model=args.model_ckpt,
        device=device,
        framework="pt",
        batch_size=args.batch_size, # This batch size is per process
    )

    texts = examples['text']
    cleaned_texts = [t.strip() if isinstance(t, str) and t.strip() else "[EMPTY_TEXT]" for t in texts]

    if not cleaned_texts or all(t == "[EMPTY_TEXT]" for t in cleaned_texts):
        num_examples_in_batch = len(texts)
        nli_scores_data = {f"nli_{hypothesis}": [0.0] * num_examples_in_batch for hypothesis in nli_hypotheses}
        return nli_scores_data

    nli_results_batch = nli_pipe_local(
        cleaned_texts,
        candidate_labels=nli_hypotheses,
    )

    nli_scores_data = {f"nli_{hypothesis}": [] for hypothesis in nli_hypotheses}
    for result_for_one_text in nli_results_batch:
        scores_map = {label: score for label, score in zip(result_for_one_text['labels'], result_for_one_text['scores'])}
        for hypothesis in nli_hypotheses:
            score = scores_map.get(hypothesis, 0.0)
            nli_scores_data[f"nli_{hypothesis}"].append(float(score))

    return nli_scores_data

# --- Function for unpacking ground truth labels (can also be parallelized) ---
def unpack_ground_truth_labels_batched(examples):
    batch_unpacked_data = {label_name: [] for label_name in labels}

    for i in range(len(examples['labels'])):
        gt_values = examples['labels'][i].tolist() if isinstance(examples['labels'][i], torch.Tensor) else examples['labels'][i]

        if len(gt_values) != len(labels):
            raise ValueError(
                f"Mismatch for example in batch: 'labels' tensor has {len(gt_values)} values, "
                f"but 'gt_label_names' has {len(labels)} names. "
                "Ensure gt_label_names has exactly 40 elements in the correct order."
            )

        for j, label_name in enumerate(labels):
            batch_unpacked_data[label_name].append(float(gt_values[j]))

    return batch_unpacked_data


# --- Main execution block (this runs only once in the parent process) ---
if __name__ == '__main__':
    # Path to your dataset
    saved_dataset_path = args.dataset_name
    print(f"Loading dataset from: {saved_dataset_path}")
    try:
        ds_with_predictions = load_from_disk(saved_dataset_path)
        print(f"Dataset loaded. Number of examples: {len(ds_with_predictions)}")
        print(f"Features: {ds_with_predictions.features}")
    except Exception as e:
        print(f"ERROR: Could not load dataset from {saved_dataset_path}. Error: {e}")
        exit(1) # Exit with an error code

    # Filter out empty or whitespace-only texts
    original_num_examples = len(ds_with_predictions)
    ds_with_predictions = ds_with_predictions.filter(
        lambda example: example['text'] is not None and len(example['text'].strip()) > 0,
        desc="Filtering out empty or whitespace texts"
    )
    filtered_num_examples = len(ds_with_predictions)
    print(f"Dataset after filtering: {filtered_num_examples} examples (Removed {original_num_examples - filtered_num_examples} empty/whitespace texts).")

    if filtered_num_examples == 0:
        print("ERROR: Dataset is empty after filtering! Cannot proceed with NLI.")
        exit(1)

    # Define new features for NLI scores
    nli_features = {}
    for hypothesis in nli_hypotheses:
        nli_features[f"nli_{hypothesis}"] = Value("float32")

    # Copy existing features and add NLI features for the NLI processing step
    final_dataset_features_with_nli = ds_with_predictions.features.copy()
    final_dataset_features_with_nli.update(nli_features)

    print("\nDefined final features for dataset after adding NLI scores:")
    print(final_dataset_features_with_nli)

    # Determine how many processes to use for GPU inference
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    actual_num_proc = min(args.num_gpus_to_use, num_gpus_available)
    if actual_num_proc == 0:
        print("WARNING: No GPUs available or requested. Running NLI on CPU (single process).")
        actual_num_proc = 1 # Fallback to single CPU process
    else:
        print(f"Adding NLI scores to the dataset using {actual_num_proc} processes on GPUs...")

    # Apply the NLI scoring function in parallel
    ds_with_all_predictions = ds_with_predictions.map(
        add_nli_scores_to_example_batched_parallel,
        batched=True,
        batch_size=args.batch_size, # This is the batch size *per process*
        features=final_dataset_features_with_nli,
        num_proc=actual_num_proc, # Crucial for multi-GPU
        with_rank=True, # Pass rank to the function for device assignment
        desc="Adding NLI Scores"
    )

    print("NLI scores added to the dataset.")
    print(ds_with_all_predictions.features)

    # Unpack ground truth labels into individual columns
    print("\nUnpacking ground truth labels into individual columns...")

    # Construct the exact target schema for the final dataset after unpacking
    # 1. Start with the features of ds_with_all_predictions
    target_features_for_unpacking = ds_with_all_predictions.features.copy()

    # 2. Explicitly remove columns that should *not* be in the final schema
    columns_to_remove_from_schema = ["labels", "input_ids", "attention_mask"]

    for col_name in columns_to_remove_from_schema:
        if col_name in target_features_for_unpacking:
            del target_features_for_unpacking[col_name]

    # 3. Add the new ground truth columns to this target schema
    # Using `labels` as you confirmed it contains your 40 ground truth labels
    for name in labels:
        target_features_for_unpacking[name] = Value("float32")

    print("\nTarget features for dataset after unpacking GT labels (after removal):")
    print(target_features_for_unpacking)

    # Unpacking can also be parallelized (usually CPU-bound)
    # You could set num_proc to os.cpu_count() here if you have many cores
    # For simplicity, we'll use the same num_proc as GPU for now if it's high enough
    # or just 1 if only CPU is available.
    unpack_num_proc = min(actual_num_proc, os.cpu_count()) if os.cpu_count() else 1
    if unpack_num_proc == 0: unpack_num_proc = 1 # Safety check
    print(f"Unpacking ground truth labels using {unpack_num_proc} processes...")

    ds_final_for_csv = ds_with_all_predictions.map(
        unpack_ground_truth_labels_batched,
        batched=True,
        batch_size=args.batch_size,
        features=target_features_for_unpacking,
        remove_columns=["labels", "input_ids", "attention_mask"], # Keep this for consistency and safety
        num_proc=unpack_num_proc, # Apply multiprocessing here too
        desc="Unpacking Ground Truth Labels"
    )

    print("Ground truth labels unpacked. Original 'labels' column removed.")
    print(ds_final_for_csv.features)

    # Sample of the final dataset (first row) before CSV export
    print("\nSample of the final dataset (first row) before CSV export:")
    if len(ds_final_for_csv) > 0:
        first_sample_final = ds_final_for_csv[0]
        # Print selected columns to verify
        print(f"tweetid: {first_sample_final.get('tweetid', 'N/A')}")
        print(f"text (first 100 chars): {first_sample_final.get('text', 'N/A')[:100]}...")
        print(f"nli_fear: {first_sample_final.get('nli_fear', 'N/A')}")
        print(f"{labels[0]}: {first_sample_final.get(labels[0], 'N/A')}")
        print(f"{labels[-1]}: {first_sample_final.get(labels[-1], 'N/A')}")
        print(f"pred_post1geo10: {first_sample_final.get('pred_post1geo10', 'N/A')}")
    else:
        print("Final dataset is empty.")

    # Save the final dataset
    print(f"\nExporting final dataset to CSV: {args.fout_nli_csv}")
    try:
        ds_final_for_csv.to_csv(args.fout_nli_csv)
        print(f"Final dataset successfully exported to CSV: {args.fout_nli_csv}")
    except Exception as e:
        print(f"ERROR: Failed to export final dataset to CSV: {e}")
