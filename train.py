from datasets import load_dataset
from transformers import (AutoTokenizer, 
                         GPT2LMHeadModel, 
                         TrainingArguments, 
                         Trainer, 
                         DataCollatorForLanguageModeling)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("text-generation/pretrained/jvc-tokenizer/")
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Add PADDING token

def tokenize_function(examples):
    return tokenizer(examples["data"], truncation=True, max_length=256)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False) # Dynamic Padding

# Dataset
dataset = load_dataset('csv', data_files='data/jvc-20k.csv') 
tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]

# Model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Training
training_args = TrainingArguments(
    output_dir="pretrained/gpt2-jvc", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model()