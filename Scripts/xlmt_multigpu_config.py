# Model name or path of model to be trained
model_ckpt = "cardiffnlp/twitter-xlm-roberta-base"

# Model name or path of model to be trained.
# model_repo_hub = "m2im/XLM-T_finetuned_violence_twitter"
model_repo_hub = "m2im/XLM-T_finetuned_violence_twitter_all_labels"

# Save dir where model repo is cloned and models updates are saved to.
# save_dir = "../../xlmt_finetuned_twitter"
save_dir = "/data4/mmendieta/models/xlmt_finetuned_twitter_all_labels"

# Name or path of training dataset.
dataset_name = "/data3/mmendieta/Violence_data/geo_corpus.0.0.1_tok_ds_xlmt_all_labels" 

# output of plots and data after training
training_output = "/data4/mmendieta/training_output/xlmt/" 

# Batch size for training.
train_batch_size = 1024

# Batch size for evaluation.
valid_batch_size = 1024

# threshold for predictions
threshold = 0.5

# Number of epochs used during training
num_epochs = 20

# Value of weight decay.
weight_decay = 0.1

# Learning rate for training.
learning_rate = 5e-5

# Learning rate scheduler type.
lr_scheduler_type = "cosine"

# Number of warmup steps in the learning rate schedule.
num_warmup_steps =0

# Sequence lengths used for training.
seq_length = 32

# Seed for replication purposes.
seed = 42

# Interval to evaluate the model and save checkpoints.
# save_checkpoint_steps = 20
 
# States path if the training should continue from a checkpoint folder. 
resume_from_checkpoint = None 

# Number of labels for the multilabel classification problem
num_labels = 40

# Push saved model to the hub.
push_to_hub = True 

# Name of the wandb project.
wandb_project = "xlmt_all_labels"

