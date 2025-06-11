import os
import pandas as pd
from glob import glob
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk
import json

print("Loading data...")
CATEGORY = "news"
def load_texts_to_dataframe(human_dir, machine_dir):
    data = []
    for file in glob(os.path.join(human_dir, CATEGORY+'*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            for entry in entries:
                human_text = entry['output'].strip()
                data.append({'text': human_text, 'label': 0})
    
    for file in glob(os.path.join(machine_dir, CATEGORY+'*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            for k, v in entries["output"].items():
                machine_text = v.strip()
                data.append({'text': machine_text, 'label': 1})

    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

human_dir = './raw_data/zh_unicode'
machine_dir = './raw_data/zh_qwen2'
df = load_texts_to_dataframe(human_dir, machine_dir)
print(f"Loaded {len(df)} samples from {human_dir} and {machine_dir}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2,
    id2label={0: "human", 1: "LLM-generated"},
    label2id={"human": 0, "LLM-generated": 1}
)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

print("Starting training...")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()

trainer.save_model("./bert_qwen_" + CATEGORY + "_classifier")
