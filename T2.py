import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from gensim.models import Word2Vec

def load_nyt(file_path):
    df = pd.read_csv(file_path)
    print("NYT columns:", df.columns.tolist())

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return df

def load_ag(file_path):
    df = pd.read_csv(file_path)
    print("AG News columns:", df.columns.tolist())

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    return df


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

def evaluate(clf, X, y, dataset_name, embedding_name):
    y_pred = clf.predict(X)

    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro"
    )

    print(f"{dataset_name} accuracy ({embedding_name}): "
          f"{accuracy:.4f}")

    print(f"{dataset_name} macro F1 ({embedding_name}): "
          f"{macro_f1:.4f}")

    print("report:")
    print(
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )

    return {
        "dataset": dataset_name,
        "embedding": embedding_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

def load_glove(glove_path, embedding_dim=100):

    embeddings = {}

    print("Loading GloVe...")

    with open(
        glove_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            values = line.rstrip().split(" ")

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype=np.float32
            )

            if len(vector) == embedding_dim:
                embeddings[word] = vector

    print(
        f"GloVe vocabulary size: "
        f"{len(embeddings)}"
    )

    return embeddings

def document_vector_glove(
    text,
    embeddings,
    embedding_dim=100
):

    tokens = tokenize(text)

    vectors = [
        embeddings[word]
        for word in tokens
        if word in embeddings
    ]

    if len(vectors) == 0:
        return np.zeros(
            embedding_dim,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    )


def transform_glove(
    texts,
    embeddings,
    embedding_dim=100
):

    return np.vstack([
        document_vector_glove(
            text,
            embeddings,
            embedding_dim
        )
        for text in texts
    ])

def document_vector_word2vec(
    text,
    model,
    embedding_dim=100
):

    tokens = tokenize(text)

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(
            embedding_dim,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    )

def transform_word2vec(
    texts,
    model,
    embedding_dim=100
):

    return np.vstack([
        document_vector_word2vec(
            text,
            model,
            embedding_dim
        )
        for text in texts
    ])

def train_logistic_regression(X_train, y_train):

    clf = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    clf.fit(
        X_train,
        y_train
    )

    return clf

def run_embedding_experiments(
    nyt_path,
    ag_path,
    glove_path
):
    nyt_df = load_nyt(nyt_path)
    ag_df = load_ag(ag_path)

    print(
        f"NYT documents: {len(nyt_df)}"
    )

    print(
        f"AG News documents: {len(ag_df)}"
    )

    X_train_text, X_temp_text, y_train, y_temp = (
        train_test_split(
            nyt_df["text"],
            nyt_df["label"],
            test_size=0.2,
            random_state=42,
            stratify=nyt_df["label"]
        )
    )

    X_val_text, X_test_text, y_val, y_test = (
        train_test_split(
            X_temp_text,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
    )

    print("\nNYT split:")
    print(
        f"Train: {len(X_train_text)}"
    )
    print(
        f"Validation: {len(X_val_text)}"
    )
    print(
        f"Test: {len(X_test_text)}"
    )

    results = []

    print("\n" + "=" * 60)
    print("Experiment 1: Pre-trained GloVe 100d")
    print("=" * 60)

    glove_embeddings = load_glove(
        glove_path,
        embedding_dim=100
    )

    X_train_glove = transform_glove(
        X_train_text,
        glove_embeddings
    )

    X_val_glove = transform_glove(
        X_val_text,
        glove_embeddings
    )

    X_test_glove = transform_glove(
        X_test_text,
        glove_embeddings
    )

    print(
        "Document vector shape:",
        X_train_glove.shape
    )

    clf_glove = train_logistic_regression(
        X_train_glove,
        y_train
    )

    results.append(
        evaluate(
            clf_glove,
            X_train_glove,
            y_train,
            "train",
            "Pre-trained GloVe 100d"
        )
    )

    results.append(
        evaluate(
            clf_glove,
            X_val_glove,
            y_val,
            "validation",
            "Pre-trained GloVe 100d"
        )
    )

    results.append(
        evaluate(
            clf_glove,
            X_test_glove,
            y_test,
            "test",
            "Pre-trained GloVe 100d"
        )
    )

    del glove_embeddings

    print("\n" + "=" * 60)
    print("Experiment 2: Word2Vec trained on AG News")
    print("=" * 60)

    print("Tokenizing AG News...")

    ag_sentences = [
        tokenize(text)
        for text in ag_df["text"]
    ]

    print("Training AG News Word2Vec...")

    w2v_ag = Word2Vec(
        sentences=ag_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        seed=42,
        epochs=5
    )

    print(
        "AG Word2Vec vocabulary size:",
        len(w2v_ag.wv)
    )

    X_train_ag = transform_word2vec(
        X_train_text,
        w2v_ag
    )

    X_val_ag = transform_word2vec(
        X_val_text,
        w2v_ag
    )

    X_test_ag = transform_word2vec(
        X_test_text,
        w2v_ag
    )

    print(
        "Document vector shape:",
        X_train_ag.shape
    )

    clf_ag = train_logistic_regression(
        X_train_ag,
        y_train
    )

    results.append(
        evaluate(
            clf_ag,
            X_train_ag,
            y_train,
            "train",
            "Word2Vec AG News 100d"
        )
    )

    results.append(
        evaluate(
            clf_ag,
            X_val_ag,
            y_val,
            "validation",
            "Word2Vec AG News 100d"
        )
    )

    results.append(
        evaluate(
            clf_ag,
            X_test_ag,
            y_test,
            "test",
            "Word2Vec AG News 100d"
        )
    )

    print("\n" + "=" * 60)
    print("Experiment 3: Word2Vec trained on NYT")
    print("=" * 60)

    print("Tokenizing NYT training documents...")

    nyt_sentences = [
        tokenize(text)
        for text in X_train_text
    ]

    print("Training NYT Word2Vec...")

    w2v_nyt = Word2Vec(
        sentences=nyt_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        seed=42,
        epochs=5
    )

    print(
        "NYT Word2Vec vocabulary size:",
        len(w2v_nyt.wv)
    )

    X_train_nyt = transform_word2vec(
        X_train_text,
        w2v_nyt
    )

    X_val_nyt = transform_word2vec(
        X_val_text,
        w2v_nyt
    )

    X_test_nyt = transform_word2vec(
        X_test_text,
        w2v_nyt
    )

    print(
        "Document vector shape:",
        X_train_nyt.shape
    )

    clf_nyt = train_logistic_regression(
        X_train_nyt,
        y_train
    )

    results.append(
        evaluate(
            clf_nyt,
            X_train_nyt,
            y_train,
            "train",
            "Word2Vec NYT 100d"
        )
    )

    results.append(
        evaluate(
            clf_nyt,
            X_val_nyt,
            y_val,
            "validation",
            "Word2Vec NYT 100d"
        )
    )

    results.append(
        evaluate(
            clf_nyt,
            X_test_nyt,
            y_test,
            "test",
            "Word2Vec NYT 100d"
        )
    )

    summary_df = pd.DataFrame(results)

    summary_df = summary_df[
        [
            "dataset",
            "embedding",
            "accuracy",
            "macro_f1"
        ]
    ]

    summary_df.to_csv(
        "nyt_embedding_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("Summary of all results")
    print("=" * 60)

    print(summary_df)

    test_results = summary_df[
        summary_df["dataset"] == "test"
    ].copy()

    print("\n" + "=" * 60)
    print("FINAL NYT TEST SET RESULTS")
    print("=" * 60)

    print(
        test_results[
            [
                "embedding",
                "accuracy",
                "macro_f1"
            ]
        ].to_string(index=False)
    )

    test_results.to_csv(
        "nyt_embedding_test_results.csv",
        index=False
    )

if __name__ == "__main__":

    nyt_path = "nyt.csv"

    ag_path = "ag.csv"

    glove_path = "glove.6B/glove.6B.100d.txt"

    run_embedding_experiments(
        nyt_path,
        ag_path,
        glove_path
    )