import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_nyt(file_path):
    df = pd.read_csv(file_path)
    print(df.columns.tolist())

    return df[["text", "label"]].dropna()

def evaluate(clf, X, y, dataset_name, vectorizer_name):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(f"{dataset_name} accuracy ({vectorizer_name}): {accuracy:.4f}")
    print(f"{dataset_name} macro F1 ({vectorizer_name}): {macro_f1:.4f}")
    print("report:")
    print(classification_report(y, y_pred, zero_division=0))

    return {
        "dataset": dataset_name,
        "vectorizer": vectorizer_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

def run_nyt(path):
    df = load_nyt(path)
    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    base_vec = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    
    results = []

    X_train_text, X_temp_text, y_train, y_temp = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    X_val_text, X_test_text, y_val, y_test = train_test_split(X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    #base_vec.fit(df["text"])
    base_vec.fit(X_train_text)
    vocab = base_vec.vocabulary_
    print(f"Vocabulary size: {len(vocab)}")

    print("="*50)
    print("binary bag-of-words")
    print("="*50)

    vec_binary = CountVectorizer(
        vocabulary=vocab,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        binary=True,
    )
    X_train_binary = vec_binary.fit_transform(X_train_text)
    X_val_binary = vec_binary.transform(X_val_text)
    X_test_binary = vec_binary.transform(X_test_text)

    clf_binary = LogisticRegression(random_state=42)
    clf_binary.fit(X_train_binary, y_train)

    results.append(
        evaluate(clf_binary, X_train_binary, y_train, "train", "binary bag-of-words")
    )
    results.append(
        evaluate(clf_binary, X_val_binary, y_val, "validation", "binary bag-of-words")
    )
    results.append(
        evaluate(clf_binary, X_test_binary, y_test, "test", "binary bag-of-words")
    )

    print("="*50)
    print("word frequency bag-of-words")
    print("="*50)

    vec_freq = CountVectorizer(
        vocabulary=vocab,
        binary=False,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train_freq = vec_freq.fit_transform(X_train_text)
    X_val_freq = vec_freq.transform(X_val_text)
    X_test_freq = vec_freq.transform(X_test_text)

    clf_freq = LogisticRegression(random_state=42)
    clf_freq.fit(X_train_freq, y_train)

    results.append(
        evaluate(clf_freq, X_train_freq, y_train, "train", "word frequency bag-of-words")
    )
    results.append(
        evaluate(clf_freq, X_val_freq, y_val, "validation", "word frequency bag-of-words")
    )
    results.append(
        evaluate(clf_freq, X_test_freq, y_test, "test", "word frequency bag-of-words")
    )

    summary_df = pd.DataFrame(results)
    summary_df = summary_df[["dataset", "vectorizer", "accuracy", "macro_f1"]]
    summary_df.to_csv("nyt_results.csv", index=False)

    print("="*50)
    print("Summary of results:")
    print("="*50)
    print(summary_df)

if __name__ == "__main__":
    nyt_path = "nyt.csv"
    run_nyt(nyt_path)
