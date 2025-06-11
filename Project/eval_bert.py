import os
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import argparse
from glob import glob
import json
import pandas as pd

parser = argparse.ArgumentParser(description="Evaluate BERT model for essay classification")
parser.add_argument("--model", type=str, default="bert_gpt_essay_classifier", help="Directory of the trained BERT model")
parser.add_argument("--eval_dataset", type=str, default="gpt_essay", help="model and category")
args = parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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

print("Loading data...")
CATEGORY = "news"
human_dir = './raw_data/zh_unicode'
machine_dir = './raw_data/zh_qwen2'
df = load_texts_to_dataframe(human_dir, machine_dir)
print(f"Loaded {len(df)} samples from {human_dir} and {machine_dir}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    id2label={0: "human", 1: "LLM-generated"},
    label2id={"human": 0, "LLM-generated": 1}
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

print("Starting evaluation...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    no_cuda=False,
    disable_tqdm=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
eval_results = trainer.evaluate()
print(f"Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")