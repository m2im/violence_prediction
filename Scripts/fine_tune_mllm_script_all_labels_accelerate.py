# import the neccesary packages
import evaluate
import numpy as np
import json
import os
import time
import torch
import transformers
import datasets
import pandas as pd
import numpy as np
import wandb
import evaluate
import labse_multigpu_config as args

from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, get_scheduler, AdamW, set_seed
from torch.utils.data import DataLoader
from huggingface_hub import Repository, get_full_repo_name
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score


gradient_accumulation_steps = args.gradient_accumulation_steps

# 1. Setup accelerator and seed
accelerator = Accelerator(log_with="wandb")
set_seed(args.seed)


# 2. Define label mappings
# List oll 40 labels
labels = ['post1geo10', 'post1geo20', 'post1geo30', 'post1geo50', 'post1geo70', 'post2geo10', 'post2geo20', 'post2geo30', 'post2geo50', 'post2geo70', 'post3geo10', 'post3geo20', 'post3geo30', 'post3geo50', 'post3geo70', 'post7geo10', 'post7geo20', 'post7geo30', 'post7geo50', 'post7geo70', 'pre1geo10', 'pre1geo20', 'pre1geo30', 'pre1geo50', 'pre1geo70', 'pre2geo10', 'pre2geo20', 'pre2geo30', 'pre2geo50', 'pre2geo70', 'pre3geo10', 'pre3geo20', 'pre3geo30', 'pre3geo50', 'pre3geo70', 'pre7geo10', 'pre7geo20', 'pre7geo30', 'pre7geo50', 'pre7geo70']

# Dynamically extract label names from dataset
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}


# 3. Define some helper functions
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    '''
    This functions differentiate the parameters that should receive weight decay
    (i.e. Biases and LayerNorm weights) are not subject to weight decay
    '''
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
           {'params': params_without_wd, 'weight_decay': 0.0}]


def get_lr():
    return optimizer.param_groups[0]['lr']


def setup_logging():
    if accelerator.is_main_process:  # we only want to setup logging once
        wandb.init(project = args.wandb_project, entity="npsdaor")
        run_name = wandb.run.name
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return run_name


def log_metrics(metrics):
    if accelerator.is_main_process:
        wandb.log(metrics)


# 4. Load dataset 
# Load the dataset from disk
tokenized_ds = load_from_disk(args.dataset_name)

# Settings
accelerator.print("[INFO] instantiating various objects required for training ...")

# Clone model repository
if accelerator.is_main_process:
    hf_repo = Repository(args.save_dir, clone_from=args.model_repo_hub)

# Logging
run_name = setup_logging()

# Checkout new branch on repo
if (accelerator.is_main_process and args.push_to_hub):
    hf_repo.git_checkout(run_name, create_branch_ok=True)


# 5. Load model and tokenizer
accelerator.print("[INFO] loading the tokenizer and the model ...")
model_ckpt = args.model_ckpt
tokenizer = AutoTokenizer.from_pretrained(model_ckpt,
                                          model_max_length=args.seq_length)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, 
                                                           num_labels = args.num_labels,
                                                           problem_type = "multi_label_classification",
                                                           id2label = id2label,
                                                           label2id = label2id)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 6. Dataloaders
# Define the Dataloaders with sample dataset
# Use a subset of 10,000 examples for debugging
# train_sample = tokenized_ds["train"].select(range(min(10000, len(tokenized_ds["train"]))))
# eval_sample = tokenized_ds["validation"].select(range(min(10000, len(tokenized_ds["validation"]))))

# train_dataloader = DataLoader(train_sample, shuffle=True, 
#                               batch_size=args.train_batch_size, collate_fn=data_collator)
# eval_dataloader = DataLoader(eval_sample,
#                              batch_size=args.valid_batch_size, collate_fn=data_collator)

# Define the Dataloaders with full dataset
train_dataloader = DataLoader(
    tokenized_ds["train"], 
    shuffle=True, 
    batch_size=args.train_batch_size, 
    collate_fn=data_collator,
    num_workers=28,
    pin_memory=True
)
eval_dataloader = DataLoader(
    tokenized_ds["validation"],
    batch_size=args.valid_batch_size, 
    collate_fn=data_collator,
    num_workers=12,
    pin_memory=True
)


# 7. Optimizer and scheduler
# Compute class weights for imbalanced multilabel data
label_tensor = torch.tensor(tokenized_ds["train"]["labels"])
label_counts = label_tensor.sum(dim=0)
total_samples = label_tensor.size(0)
neg_counts = total_samples - label_counts
pos_weight = (neg_counts / (label_counts + 1e-6)).to(torch.float32)  # avoid div by 0

# Define the optimizer
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)

# Define the learning rate scheduler
num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader) 
accelerator.print(num_training_steps)
lr_scheduler = get_scheduler(
    name = args.lr_scheduler_type,
    optimizer = optimizer,
    num_warmup_steps = args.num_warmup_steps,
    num_training_steps = num_training_steps
) 


# 8. Prepare everything with our `accelerator`.
model, optimizer, train_dl, eval_dl = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
    )

# 9. Evaluate function
def evaluate_fn(args):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    batch_count = 0

    accelerator.print(f"[DEBUG] Starting evaluation... Number of batches: {len(eval_dl)}")

    for batch_idx, batch in enumerate(eval_dl):
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, batch["labels"].float())
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > args.threshold).int()

        gathered_preds = accelerator.gather(preds).cpu().numpy()
        gathered_probs = accelerator.gather(probs).cpu().numpy()
        gathered_labels = accelerator.gather(batch["labels"]).cpu().numpy()

        all_preds.append(gathered_preds)
        all_probs.append(gathered_probs)
        all_labels.append(gathered_labels)

        batch_count += 1
        # accelerator.print(f"[DEBUG] Processed batch {batch_idx+1}/{len(eval_dl)}")
        
        if batch_idx == len(eval_dl) - 1 or batch_idx % 100 == 0:
            accelerator.print(f"[INFO] Processed eval batch {batch_idx+1}/{len(eval_dl)}")

    if batch_count == 0:
        raise RuntimeError("No evaluation batches were processed. Check dataloader.")

    np_preds = np.vstack(all_preds)
    np_probs = np.vstack(all_probs)
    np_labels = np.vstack(all_labels)

    try:
        roc_auc_micro = roc_auc_score(np_labels, np_probs, average="micro")
        roc_auc_weighted = roc_auc_score(np_labels, np_probs, average="weighted")
        precision_micro = precision_score(np_labels, np_preds, average="micro", zero_division=0)
        precision_weighted = precision_score(np_labels, np_preds, average="weighted", zero_division=0)
        recall_micro = recall_score(np_labels, np_preds, average="micro", zero_division=0)
        recall_weighted = recall_score(np_labels, np_preds, average="weighted", zero_division=0)
        f1_micro = f1_score(np_labels, np_preds, average="micro", zero_division=0)
        f1_weighted = f1_score(np_labels, np_preds, average="weighted", zero_division=0)
        conf_matrices = multilabel_confusion_matrix(np_labels, np_preds)
        
        # Per-label metrics
        per_label_metrics = {
            label: {
                "roc_auc": roc_auc_score(np_labels[:, idx], np_probs[:, idx]),
                "precision": precision_score(np_labels[:, idx], np_preds[:, idx], zero_division=0),
                "recall": recall_score(np_labels[:, idx], np_preds[:, idx], zero_division=0),
                "f1": f1_score(np_labels[:, idx], np_preds[:, idx], zero_division=0)
            }
            for idx, label in enumerate(labels)
        }
               
    except ValueError as e:
        accelerator.print(f"[ERROR] Metric computation failed: {e}")
        return {}, torch.tensor(total_loss / batch_count), {}

    # Save confusion matrices
    output_dir = args.training_output
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "confusion_matrices.npy"), conf_matrices)     

    metrics_dict = {
        "roc_auc_micro_score": roc_auc_micro,
        "roc_auc_weighted_score": roc_auc_weighted,
        "precision_micro": precision_micro,
        "precision_weighted": precision_weighted,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
    }

    return metrics_dict, torch.tensor(total_loss / batch_count), per_label_metrics

# 10. Training loop
# load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "":
        checkpoint_path = args.resume_from_checkpoint
    else:
        # Auto-detect most recent epoch folder
        dirs = [f.name for f in os.scandir(args.save_dir) if f.is_dir() and "epoch_" in f.name]
        dirs.sort(key=os.path.getctime)
        checkpoint_path = os.path.join(args.save_dir, dirs[-1])
    
    accelerator.print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)
    resume_epoch = int(os.path.basename(checkpoint_path).replace("epoch_", ""))


# Training loop
accelerator.print("[INFO] training starts...")
progress_bar = tqdm(range(num_epochs), disable = not accelerator.is_local_main_process)


t_start = time.time()
old_roc = 0

for epoch in range(num_epochs):
    if args.resume_from_checkpoint and epoch < resume_epoch:
            continue   # we need to skip steps until we reach the resumed epoch
            
    model.train()
    accelerator.print(f"[INFO] Starting epoch {epoch}")
    
    for step, batch in enumerate(train_dl):
        if step % 1000 == 0 or step == len(train_dl) - 1:
            accelerator.print(f"[DEBUG] - Processing step {step+1}/{len(train_dl)} on device {accelerator.device}")         
            
        outputs = model(**batch)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fn(logits, batch["labels"].float())
        loss = loss/gradient_accumulation_steps  # scale loss
        lr = get_lr()
        accelerator.backward(loss)
        
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dl):
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    accelerator.print('[INFO] evaluating and saving model checkpoint...')      
    metrics_dict, eval_loss, per_label_metrics = evaluate_fn(args)
    elapsed_time = time.time() - t_start
    
    # Log to wandb
    log_metrics({**metrics_dict, "epoch": epoch, "elapsed_time": elapsed_time,
                 "loss/train": loss.item() * gradient_accumulation_steps, # unscale for logging 
                 "loss/eval": eval_loss.item(), "lr": lr})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Save and push only the model that improves roc_auc_score on the validation set
    if metrics_dict['roc_auc_micro_score'] > old_roc:
        save_dir = os.path.join(os.path.join(args.save_dir, run_name), f"epoch_{epoch}")
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
            if args.push_to_hub:
                hf_repo.push_to_hub(commit_message=f"epoch_{epoch}")
                
            # Write per-label metrics JSON on improvement
            output_dir = args.training_output
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "per_label_metrics.json")
            with open(os.path.join(output_dir, "per_label_metrics.json"), "w") as f:
                json.dump(per_label_metrics, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            accelerator.print(f"[INFO] Per-label metrics written to: {json_path}")
           
        old_roc = metrics_dict['roc_auc_micro_score']
    
    progress_bar.update(1)

    # Console summary
    accelerator.print(
        f"[epoch {epoch}] train/loss: {loss.item():.4f} | "
        f"eval/loss: {eval_loss.item():.4f} | "
        f"roc_auc_micro: {metrics_dict['roc_auc_micro_score']:.4f} | "
        f"roc_auc_weighted: {metrics_dict['roc_auc_weighted_score']:.4f} | "
        f"precision_micro: {metrics_dict['precision_micro']:.4f} | "
        f"precision_weighted: {metrics_dict['precision_weighted']:.4f} | "
        f"recall_micro: {metrics_dict['recall_micro']:.4f} | "
        f"recall_weighted: {metrics_dict['recall_weighted']:.4f} | "
        f"f1_micro: {metrics_dict['f1_micro']:.4f} | "
        f"f1_weighted: {metrics_dict['f1_weighted']:.4f} | "
        f"elapsed_time: {elapsed_time:.2f}s"
    )



