import sys 
from typing import List, Any, Dict 
from datasets import Dataset, load_dataset 
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers.data import *
from transformers import TrainingArguments, Trainer 
from peft import get_peft_model, LoraConfig, TaskType
import evaluate 
import numpy as np 
import pandas as pd 
from modeling_llama import LlamaForSequenceClassification 


train_dataset = '../data/java_train.csv'
test_dataset = '../data/java_test.csv'
# train_dataset = "../data/flaky_C_train.csv"
# test_dataset = '../data/flaky_C_test.csv'

model = sys.argv[1]

if not sys.argv:
    print('Please input the model you want to use: llama2-13b, llama2-7b codellama-7b or mistral-7b')
    sys.exit()


epoch = 10
batch_size = 8
lora_r = 12
max_length=512
learning_rate=5e-5

if model == "llama2-7b":
    model_id = "NousResearch/Llama-2-7b-hf"
elif model == "llama2-13b":
    model_id = "NousResearch/Llama-2-13b-hf"
elif model == "codellama-7b":
    model_id = "codellama/CodeLlama-7b-hf"
elif model == "codellama-13b":
    model_id = 'codellama/CodeLlama-13b-hf'
elif model == 'mistral-7b':
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

test_name = 'test'
text_name = None

if train_dataset == '../data/java_train.csv':
    id2label = {0:'async wait',1:'concurrency',2:'unordered collections',3:'test case timeout',4:'time',
            5:'test order dependency',6:'floating point operations', 7:'randomness',8:'network',9:'i_o',
            10:'resource leak', 11:'too restrictive range', 12:'platform dependency'}
    label2id = {v:k for k,v in id2label.items()}
    ds = load_dataset("csv", data_files=train_dataset)
    text_name = 'text'
    print(ds['train'].features)
elif train_dataset == "../data/flaky_C_train1.csv":
    label2id =  {'async wait':0,'concurrency':1,'time':3,'insufficient assertion': 
            4,'hash operation': 5,'Float point': 6,'test data sensitive':2,'IO':7, 'unordered collection':8, 'randomness':9}
    id2label = {v:k for k,v in label2id.items()}
    ds = load_dataset("csv", data_files=test_dataset)
    text_name = 'text'    

if test_dataset == "../data/java_test.csv":
    id2label = {0:'async wait',1:'concurrency',2:'unordered collections',3:'test case timeout',4:'time',
            5:'test order dependency',6:'floating point operations', 7:'randomness',8:'network',9:'i_o',
            10:'resource leak', 11:'too restrictive range', 12:'platform dependency'}
    label2id = {v:k for k,v in id2label.items()}
    test_ds = load_dataset("csv", data_files=test_dataset)
    text_name = 'text'
    print(test_ds['train'].features)

if test_dataset == "../data/flaky_C_test1.csv":
    label2id =  {'async wait':0,'concurrency':1,'time':3,'insufficient assertion': 
            4,'hash operation': 5,'Float point': 6,'test data sensitive':2,'IO':7, 'unordered collection':8, 'randomness':9}
    id2label = {v:k for k,v in label2id.items()}
    test_ds = load_dataset("csv", data_files=test_dataset)
    text_name = 'text'



accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")



if model_id == "mistralai/Mistral-7B-Instruct-v0.2" or model_id == "codellama/CodeLlama-7b-hf" or model_id == 'codellama/CodeLlama-13b-hf':
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        cache_dir='../Models/',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
else:
    model = LlamaForSequenceClassification.from_pretrained(
        model_id, 
        cache_dir='../Models/',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False,
                         r=lora_r,
                         lora_alpha=32,
                         lora_dropout=0.1)
model=get_peft_model(model, peft_config)

model.print_trainable_parameters()
for name, param in model.named_parameters():
    if 'embed' in name:
        print("******************")
        param.requires_grad=False
    # print(name)


model.print_trainable_parameters()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    precision = precision_metric.compute(predictions=predictions, references=labels, average=None)['precision']
    recall = recall_metric.compute(predictions=predictions, references=labels, average=None)['recall']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
 
    print('\nprecision:', precision)
    print('\n')
    print('recall: ', recall)
    return {"f1-score": f1, 'accuracy': accuracy}


def preprocessing_function(examples):
    d = examples[text_name]
    return tokenizer(d, padding='longest', max_length=max_length, truncation=True,)

tokenized_dataset = ds.map(preprocessing_function, batched=True)
test_ds = test_ds.map(preprocessing_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./java-llama-new/",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()