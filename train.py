import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"Allocated GPU Memory: {allocated:.2f} GB")
        print(f"Reserved GPU Memory: {reserved:.2f} GB")

# Load the dataset from JSONL
dataset = load_dataset('json', data_files={'train': '/home/hurairah/Documents/AI-Training/subset.jsonl'})
train_dataset = dataset['train']

# Load model and tokenizer from Hugging Face
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Clear CUDA cache
torch.cuda.empty_cache()

# Training arguments with further reduced settings
training_args = TrainingArguments(
    output_dir="/home/hurairah/Documents/AI-Training/results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Further reduced batch size
    gradient_accumulation_steps=1,  # Further reduced gradient accumulation steps
    fp16=False,  # Disable mixed precision to see if it helps
    logging_dir="/home/hurairah/Documents/AI-Training/logs",
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=100,  # Less frequent logging to save resources
    report_to="tensorboard",
    dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
    gradient_checkpointing=True,  # Enable gradient checkpointing
    max_seq_length=16,  # Further reduce sequence length
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Start training and print GPU memory usage
print_gpu_memory()  # Check memory before starting training
trainer.train()
print_gpu_memory()  # Check memory after training
