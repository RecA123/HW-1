import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import numpy as np
from datasets import Dataset
from typing import Any, cast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
RANDOM_SEED = 42

def load_nyt(file_path):
    df = pd.read_csv(file_path)
    print(df.columns.tolist())

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return df

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

def run_bert(file_path):
    set_seed(RANDOM_SEED)

    df = load_nyt(file_path)
    print(f"Loaded {len(df)} samples from {file_path}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X_train_text, X_temp_text, y_train, y_temp = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    X_val_text, X_test_text, y_val, y_test = train_test_split(X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("\n" + "=" * 60)
    print("Dataset Split")
    print("=" * 60)

    print(f"Train:      {len(X_train_text)}")
    print(f"Validation: {len(X_val_text)}")
    print(f"Test:       {len(X_test_text)}")

    label_names = sorted(df["label"].unique())

    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    print("\nLabel mapping:")

    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    train_df = pd.DataFrame({"text": X_train_text, "label": [label2id[x] for x in y_train.values]})
    val_df = pd.DataFrame({"text": X_val_text, "label": [label2id[x] for x in y_val.values]})
    test_df = pd.DataFrame({"text": X_test_text, "label": [label2id[x] for x in y_test.values]})

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_names), label2id=label2id, id2label=id2label)

    training_args = TrainingArguments(
        output_dir="./bert_nyt_results",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=RANDOM_SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_results = trainer.evaluate(eval_dataset=cast(Any, val_dataset))
    print("\nValidation Results:")
    print(val_results)

    test_output = trainer.predict(test_dataset=cast(Any, test_dataset))
    raw_predictions = test_output.predictions

    if raw_predictions is None:
        raise ValueError("Model predictions are missing.")

    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]

    test_logits = np.asarray(raw_predictions)
    test_predictions = np.argmax(test_logits, axis=-1)

    raw_labels = test_output.label_ids

    if raw_labels is None:
        raise ValueError("Test labels are missing.")

    if isinstance(raw_labels, tuple):
        raw_labels = raw_labels[0]

    test_labels = np.asarray(raw_labels)
    test_labels = test_labels.reshape(-1)

    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_macro_f1 = f1_score(test_labels, test_predictions, average="macro")
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy}")
    print(f"Macro F1: {test_macro_f1}")
    print("\nClassification Report:")

    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=list(range(len(label_names))),
            target_names=[
                id2label[i]
                for i in range(len(label_names))
            ],
            digits=4,
            zero_division=0
        )
    )

    result_df = pd.DataFrame([
        {
            "model": "BERT-base-uncased",
            "max_length": MAX_LENGTH,
            "epochs": NUM_EPOCHS,
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1
        }
    ])

    result_df.to_csv(
        "nyt_bert_test_results.csv",
        index=False
    )

    print("\nResults saved to:")
    print("nyt_bert_test_results.csv")

    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)

    print(result_df.to_string(index=False))

if __name__ == "__main__":
    nyt_path = "nyt.csv"
    run_bert(nyt_path)