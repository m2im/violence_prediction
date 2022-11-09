# import the neccesary packages
import os
import time
import torch
import transformers
import datasets
import pandas as pd
import numpy as np
import wandb
import evaluate
import small_labse_multigpu_config as args

from pathlib import Path
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_from_disk
from datasets import Value, Sequence
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, get_scheduler, AdamW, set_seed
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from huggingface_hub import Repository, get_full_repo_name


# Define some helper functions
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    '''
    This functions differentiate he parameters that should receive weight decay
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
        # accelerator.init_trackers(project_name, vars(args))
        # run_name = accelerator.trackers[0].run.name
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


# Load required metrics
accuracy_metric = evaluate.load("accuracy")
roc_auc_metric = evaluate.load("roc_auc", "multilabel")
precision_micro_metric = evaluate.load("precision")
precision_weighted_metric = evaluate.load("precision")
recall_micro_metric = evaluate.load("recall")
recall_weighted_metric = evaluate.load("recall")
f1_micro_metric = evaluate.load("f1")
f1_weighted_metric = evaluate.load("f1")


def evaluate_fn(args):
    model.eval()
    for batch in eval_dl:
        with torch.no_grad():
            outputs = model(**batch)
        eval_loss = outputs.loss
        logits = outputs.logits
        pred_prob = torch.sigmoid(logits)
        preds = (pred_prob > args.threshold)*1
            
        for references, predictions in zip (batch["labels"], preds):
            accuracy_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            roc_auc_metric.add_batch(prediction_scores=accelerator.gather(pred_prob), 
                                  references=accelerator.gather(batch["labels"]))
            precision_micro_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            precision_weighted_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            recall_micro_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            recall_weighted_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            f1_micro_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
            f1_weighted_metric.add_batch(predictions=accelerator.gather(predictions), 
                references=accelerator.gather(references))
   
    accuracy_res = accuracy_metric.compute()
    roc_auc_res = roc_auc_metric.compute(average="micro")
    precision_micro_res = precision_micro_metric.compute(average="micro")
    precision_weighted_res = precision_weighted_metric.compute(average="weighted")
    recall_micro_res = recall_micro_metric.compute(average="micro")
    recall_weighted_res = recall_weighted_metric.compute(average="weighted")
    f1_micro_res = f1_micro_metric.compute(average="micro")
    f1_weighted_res = f1_weighted_metric.compute(average="weighted")

    return accuracy_res, roc_auc_res, precision_micro_res, precision_weighted_res, recall_micro_res, recall_weighted_res, f1_micro_res, f1_weighted_res, eval_loss


# Load the dataset from disk
tokenized_ds = load_from_disk(args.dataset_name)

# Settings
print("[INFO] instantiating various objects required for training ...")

# Accelerator
accelerator = Accelerator(log_with="wandb")
set_seed(args.seed)

# Clone model repository
if accelerator.is_main_process:
    hf_repo = Repository(args.save_dir, clone_from=args.model_repo_hub)

# Logging
run_name = setup_logging()

# Checkout new branch on repo
if accelerator.is_main_process:
    hf_repo.git_checkout(run_name, create_branch_ok=True)


# Load model and tokenizer
print("[INFO] loading the tokenizer and the model ...")
id2label = {0: 'post7geo10', 1: 'post7geo30', 2: 'post7geo50', 3: 'pre7geo10', 4: 'pre7geo30', 5: 'pre7geo50'}
label2id = {'post7geo10': 0, 'post7geo30': 1, 'post7geo50': 2, 'pre7geo10': 3, 'pre7geo30': 4, 'pre7geo50': 5}
model_ckpt = args.model_ckpt
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, 
	model_max_length=args.seq_length)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, 
	num_labels = args.num_labels,
	problem_type = "multi_label_classification",
	id2label = id2label,
	label2id = label2id)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Select a subsample for testing purposes
train_tokenized_ds = tokenized_ds["train"].select(range(20000))
validation_tokenized_ds = tokenized_ds["validation"].select(range(5000))


# Define the Dataloaders
# train_dataloader = DataLoader(tokenized_ds["train"], shuffle=True, 
#                               batch_size=args.batch_size, collate_fn=data_collator)
# eval_dataloader = DataLoader(tokenized_ds["validation"],
#                             batch_size=args.batch_size, collate_fn=data_collator)
train_dataloader = DataLoader(train_tokenized_ds, shuffle=True, 
	batch_size=args.train_batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(validation_tokenized_ds,
	batch_size=args.valid_batch_size, collate_fn=data_collator)


# Define the optimizer
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)

# Define the learning rate scheduler
num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name = args.lr_scheduler_type,
    optimizer = optimizer,
    num_warmup_steps = args.num_warmup_steps,
    num_training_steps = num_training_steps
) 


# Prepare everything with our `accelerator`.
model, optimizer, train_dl, eval_dl = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
    )


# load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = [f.name for f in os.scandir(args.save_dir) if f.is_dir() and "step" in str(f)]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    # Extract the step of the checkpoint to continue from there
    training_difference = os.path.splitext(path)[0]
    resume_step = int(training_difference.replace("step_", ""))


# Training loop
print("[INFO] training starts...")
progress_bar = tqdm(range(num_training_steps), disable = not accelerator.is_local_main_process)

model.train()
completed_steps = 0
t_start = time.time()
for epoch in range(num_epochs):
    for batch in train_dl:
        if args.resume_from_checkpoint and completed_steps < resume_step:
            continue   # we need to skip steps until we reach the resumed step
        outputs = model(**batch)
        loss = outputs.loss
        lr = get_lr()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1

        if completed_steps % args.save_checkpoint_steps == 0:
            elapsed_time = time.time() - t_start
            acc, roc, precMi, precW, recMi, recW, f1Mi, f1W, eval_loss = evaluate_fn(args)        
            log_metrics({"steps": completed_steps, 
                "elapsed_time": elapsed_time,
                "loss/train": loss, 
                "loss/eval": eval_loss,
                "lr": lr,
                "accuracy": acc,
                "roc_auc_score": roc,
                "precision_micro": precMi,
                "precision_weighted": precW,
                "recall_micro": recMi,
                "recall_weighted": recW,
                "f1_micro": f1Mi,
                "f1_weighted": f1W})
            accelerator.wait_for_everyone()
            save_dir = os.path.join(args.save_dir, f"step_{completed_steps}")
            accelerator.save_state(save_dir)
            
            if accelerator.is_main_process:
                hf_repo.push_to_hub(commit_message=f"step {completed_steps}")
        
        progress_bar.update(1)

        if completed_steps >= num_training_steps:
            break
        

# Evaluate and save the last checkpoint
acc, roc, precMi, precW, recMi, recW, f1Mi, f1W, eval_loss = evaluate_fn(args) 
log_metrics({"steps": completed_steps,
    "loss/train": loss,
    "loss/eval": eval_loss,
    "lr": lr,
    "accuracy": acc,
    "roc_auc_score": roc,
    "precision_micro": precMi,
    "precision_weighted": precW,
    "recall_micro": recMi,
    "recall_weighted": recW,
    "f1_micro": f1Mi,
    "f1_weighted": f1W})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(args.save_dir, save_function=accelerator.save)
save_dir = os.path.join(args.save_dir, f"step_{completed_steps}")
accelerator.save_state(save_dir)
if accelerator.is_main_process:
    hf_repo.push_to_hub(commit_message = "final model")


# # Print results
# print("Accuracy: ", accuracy_res)
# print("roc_auc: ", roc_auc_res)
# print("Precision_micro: ", precision_micro_res)
# print("Precision_weighted: ", precision_weighted_res)
# print("Recall_micro: ", recall_micro_res)
# print("Recall_weighted: ", recall_weighted_res)
# print("F1_micro: ", f1_micro_res)
# print("F1_weighted: ", f1_weighted_res)
