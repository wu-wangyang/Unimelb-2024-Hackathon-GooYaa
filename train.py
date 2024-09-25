import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# 1. Load the product-to-industry mapping table
data = pd.read_csv('./data/dataset.csv')

# Convert industry categories to numeric labels
industry_labels = {industry: idx for idx, industry in enumerate(data['Industry'].unique())}
data['label'] = data['Industry'].map(industry_labels)

# Convert the dataset to the Hugging Face Dataset format
dataset = Dataset.from_pandas(data)

# 2. Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(industry_labels))

# Print device information, ensuring the use of GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Move the model to GPU or CPU
model.to(device)

# Convert product names into model input
def tokenize_function(examples):
    return tokenizer(examples['Product Name'], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# 4. Set training parameters
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save results
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=4,  # Batch size for training per device
    per_device_eval_batch_size=4,  # Batch size for evaluation per device
    num_train_epochs=3,  # Batch size for evaluation per device
    weight_decay=0.01,  # Batch size for evaluation per device
    fp16=True,  # Enable mixed precision training
)

# 5. Create a Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()

# 5. Create a Trainer and start training
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
