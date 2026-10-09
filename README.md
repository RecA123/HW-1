# NLP Homework 1

本仓库为自然语言处理课程第一次作业，实现并比较不同文本表示方法在新闻文本分类任务中的性能，包括 Bag of Words、Word Embedding 和 BERT Fine-tuning。

主要使用 New York Times（NYT）新闻数据集进行文本分类，并使用 AG News 数据集训练 Word2Vec 词向量。

## 1. Project Structure

```text
HW-1/
├── T1.py                           # Task 1: Bag of Words
├── T2.py                           # Task 2: Word Embeddings
├── T3.py                           # Task 3: BERT Fine-tuning
├── run.sh                          # Git Bash 运行脚本
├── run.ps1                         # PowerShell 运行脚本（可选）
├── requirements.txt                # Python 依赖（如已创建）
├── README.md
├── .gitignore
│
├── nyt.csv                          # NYT 数据集（需自行准备）
├── ag.csv                           # AG News 数据集（需自行准备）
├── glove.6B/
│   └── glove.6B.100d.txt            # GloVe 100维词向量
│
├── nyt_results.csv                  # Task 1 实验结果
├── nyt_embedding_results.csv        # Task 2 全部实验结果
├── nyt_embedding_test_results.csv   # Task 2 测试集结果
└── nyt_bert_test_results.csv        # Task 3 测试集结果
```

注意：大型数据集、GloVe 词向量和 BERT 训练检查点不上传至 GitHub，需要在本地准备。

## 2. Environment Setup

推荐使用 Python 3.10 或 3.11，并通过 Conda 管理环境。

### 2.1 创建环境

在 Windows PowerShell 或 Anaconda Prompt 中执行：

```powershell
conda create -n nlp_hw1 python=3.10 -y
conda activate nlp_hw1
```

### 2.2 安装依赖

```powershell
python -m pip install numpy pandas scipy scikit-learn gensim
python -m pip install torch transformers datasets accelerate
```

BERT 训练建议使用支持 CUDA 的 NVIDIA GPU。PyTorch 的安装方式可参考官方文档：

https://pytorch.org/get-started/locally/

## 3. Dataset Preparation

### 3.1 NYT Dataset

NYT 数据集包含两列：

| Column  | Description |
| ------- | ----------- |
| `text`  | 新闻文本        |
| `label` | 新闻类别        |

所有分类任务均使用 NYT 数据集。

数据集划分：

| Dataset    | Ratio |
| ---------- | ----- |
| Training   | 80%   |
| Validation | 10%   |
| Test       | 10%   |

代码使用 `random_state=42`，并按照标签进行分层抽样（Stratified Split），以保证不同实验的数据划分一致。

### 3.2 AG News Dataset

AG News 数据集用于 Task 2 中 Word2Vec 词向量的训练。

### 3.3 Pre-trained GloVe

Task 2 使用预训练的 GloVe 100-dimensional word vectors。

下载地址：

https://nlp.stanford.edu/data/glove.6B.zip

下载并解压后，将 `glove.6B.100d.txt` 放在：

```text
HW-1/glove.6B/glove.6B.100d.txt
```

## 4. Task 1: Bag of Words

运行文件：`T1.py`

本任务使用 Logistic Regression 分类器，对比两种 Bag-of-Words 文本表示方法。

### 4.1 Binary Bag of Words

只记录单词是否在文档中出现：

* 1：单词出现
* 0：单词未出现

使用 `CountVectorizer(binary=True)` 实现。

### 4.2 Word Frequency

统计每个单词在文档中出现的次数。

使用 `CountVectorizer(binary=False)` 实现。

两种方法均使用从训练集构建的词表，避免测试集信息泄漏。

### 4.3 Run Task 1

```powershell
python T1.py
```

输出文件：

```text
nyt_results.csv
```

包含训练集、验证集和测试集的 Accuracy 与 Macro-F1。

## 5. Task 2: Word Embeddings

运行文件：`T2.py`

本任务使用 100 维词向量表示文档，并使用 Logistic Regression 进行文本分类。

文档向量由所有有效单词的词向量取平均得到。

### 5.1 Pre-trained GloVe

使用预训练的 `glove.6B.100d.txt` 获取单词向量，并计算平均文档向量。

### 5.2 Word2Vec Trained on AG News

使用 AG News 文本训练 Word2Vec：

* Vector size: 100
* Window size: 5
* Min count: 1
* Architecture: Skip-gram
* Epochs: 5

使用训练好的词向量表示 NYT 文档。

### 5.3 Word2Vec Trained on NYT

仅使用 NYT 训练集文本训练 Word2Vec，参数与 AG News Word2Vec 相同。

### 5.4 Run Task 2

```powershell
python T2.py
```

输出文件：

```text
nyt_embedding_results.csv
nyt_embedding_test_results.csv
```

分别保存全部数据划分的实验结果，以及测试集上的三组词向量实验结果。

## 6. Task 3: BERT Fine-tuning

运行文件：`T3.py`

使用 Hugging Face Transformers 对预训练 BERT 模型进行微调。

预训练模型：

https://huggingface.co/google-bert/bert-base-uncased

训练配置：

| Parameter               | Value                         |
| ----------------------- | ----------------------------- |
| Model                   | google-bert/bert-base-uncased |
| Maximum sequence length | 64                            |
| Epochs                  | 3                             |
| Learning rate           | 2e-5                          |
| Training batch size     | 16                            |
| Evaluation batch size   | 32                            |
| Random seed             | 42                            |
| Evaluation strategy     | Every epoch                   |
| Checkpoint strategy     | Every epoch                   |

模型通过 `Trainer` 完成训练，并在 NYT Test Set 上进行最终评估。

### 6.1 Run Task 3

```powershell
python T3.py
```

输出文件：

```text
nyt_bert_test_results.csv
```

训练检查点保存在：

```text
bert_nyt_results/
```

首次运行时需要从 Hugging Face 下载预训练模型。

## 7. Running the Experiments

所有命令均需在项目根目录 `HW-1` 下执行。

### Git Bash

```bash
bash run.sh 1
bash run.sh 2
bash run.sh 3
bash run.sh
```

## 8. Evaluation Metrics

所有分类实验均使用以下评价指标：

**Accuracy**

正确预测的样本数占总样本数的比例。

**Macro-F1**

先分别计算每个类别的 F1-score，再对所有类别取算术平均。

最终性能统一在 NYT Test Set 上报告。

## 9. Experiment Results and Analysis

本项目在 NYT 新闻分类数据集上，对比了 Bag of Words、Word Embedding 和 BERT 三类文本分类方法。

所有模型均使用相同的数据划分，并通过 **准确率（Accuracy）** 和 **宏平均 F1 值（Macro-F1）** 进行评估。

### 9.1 数据集划分

NYT 数据集共包含 **11,519 条新闻文本**，分为 `business`、`politics` 和 `sports` 三个类别。

| 数据集 |   样本数量 |   比例 |
| --- | -----: | ---: |
| 训练集 |  9,215 |  80% |
| 验证集 |  1,152 |  10% |
| 测试集 |  1,152 |  10% |
| 总计  | 11,519 | 100% |

数据集使用 `random_state=42` 进行分层随机划分，确保各实验使用相同的训练集、验证集和测试集。

### 9.2 各模型测试集结果对比

| 任务     | 文本表示方法                            |   Accuracy |   Macro-F1 |
| ------ | --------------------------------- | ---------: | ---------: |
| Task 1 | Binary Bag of Words（二值词袋）         |     0.9887 |     0.9718 |
| Task 1 | Word Frequency Bag of Words（词频词袋） | **0.9896** | **0.9743** |
| Task 2 | Pre-trained GloVe 100d            |     0.9852 |     0.9646 |
| Task 2 | Word2Vec（AG News 训练）              |     0.9766 |     0.9416 |
| Task 2 | Word2Vec（NYT 训练）                  |     0.9818 |     0.9542 |
| Task 3 | BERT-base-uncased                 |     0.9818 |     0.9591 |

### 9.3 Task 1：Bag of Words 实验结果

本任务使用逻辑回归（Logistic Regression）作为分类器，比较两种词袋表示方式。

| 方法                 | 验证集 Accuracy | 验证集 Macro-F1 | 测试集 Accuracy | 测试集 Macro-F1 |
| ------------------ | -----------: | -----------: | -----------: | -----------: |
| Binary BoW         |       0.9809 |       0.9551 |       0.9887 |       0.9718 |
| Word Frequency BoW |       0.9835 |       0.9626 |   **0.9896** |   **0.9743** |

### 9.4 Task 2：Word Embedding 实验结果

本任务使用三种 100 维词向量，通过平均池化生成文档向量，再使用逻辑回归进行分类。

| 词向量方法                  | 测试集 Accuracy | 测试集 Macro-F1 |
| ---------------------- | -----------: | -----------: |
| Pre-trained GloVe 100d |   **0.9852** |   **0.9646** |
| Word2Vec（AG News）      |       0.9766 |       0.9416 |
| Word2Vec（NYT）          |       0.9818 |       0.9542 |

### 9.5 Task 3：BERT 微调实验结果

本任务使用预训练模型 `google-bert/bert-base-uncased` 进行新闻文本分类。

主要训练参数如下：

| 参数     | 设置                |
| ------ | ----------------- |
| 预训练模型  | bert-base-uncased |
| 最大序列长度 | 64                |
| 训练轮数   | 3                 |
| 学习率    | 2e-5              |
| 训练批次大小 | 16                |
| 评估批次大小 | 32                |
| 随机种子   | 42                |

**验证集结果：**

| Epoch |   Accuracy |   Macro-F1 |
| ----- | ---------: | ---------: |
| 1     |     0.9688 |     0.9315 |
| 2     | **0.9766** | **0.9524** |
| 3     |     0.9757 |     0.9498 |

**最终测试集结果：**

| 指标       |     结果 |
| -------- | -----: |
| Accuracy | 0.9818 |
| Macro-F1 | 0.9591 |

### 9.6 实验结果文件

所有实验结果均保存在 CSV 文件中，便于复现和进一步分析。

| 文件名                              | 内容                        |
| -------------------------------- | ------------------------- |
| `nyt_results.csv`                | Task 1 词袋模型实验结果           |
| `nyt_embedding_results.csv`      | Task 2 词向量模型训练集、验证集及测试集结果 |
| `nyt_embedding_test_results.csv` | Task 2 词向量模型测试集结果         |
| `nyt_bert_test_results.csv`      | Task 3 BERT 模型测试集结果       |

