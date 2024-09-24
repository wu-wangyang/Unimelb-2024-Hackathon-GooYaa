import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# 1. 加载产品与行业的映射表
data = pd.read_csv('./data/dataset.csv')

# 将类别转换为数字标签
industry_labels = {industry: idx for idx, industry in enumerate(data['Industry'].unique())}
data['label'] = data['Industry'].map(industry_labels)

# 转换为 Hugging Face 的 Dataset 格式
dataset = Dataset.from_pandas(data)

# 2. 加载BERT模型和Tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(industry_labels))

# 打印设备信息，确保使用GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 将模型移动到GPU或CPU
model.to(device)

# 将产品名称转换为模型输入
def tokenize_function(examples):
    return tokenizer(examples['Product Name'], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. 将数据集分为训练集和验证集
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 输出结果的文件夹
    evaluation_strategy="epoch",  # 每个epoch评估一次
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=4,  # 每个设备上的训练批大小
    per_device_eval_batch_size=4,  # 每个设备上的验证批大小
    num_train_epochs=3,  # 训练轮数
    weight_decay=0.01,  # 权重衰减
    fp16=True,  # 启用混合精度训练
)

# 5. 创建Trainer并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 开始训练
trainer.train()

# 6. 保存训练好的模型和tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
