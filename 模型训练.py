#!/usr/bin/env python
# coding: utf-8

# 基于ERNIE Tiny的中文情感分析系统

# In[1]:


# 解压数据集
get_ipython().run_line_magic('cd', '/home/aistudio/data/data100731/')
get_ipython().system('unzip shuju.zip')


# In[2]:


# 使用pandas读取数据集
import pandas as pd
data = pd.read_csv('OCEMOTION.csv', sep='\t',header=None)
# 由于该数据集没有列名，故需要为其添加上列名以便于更好处理
data.columns = ["id", "text_a", "label"]


# In[3]:


# 查看数据前5条内容
data.head()


# In[4]:


# 查看数据文件信息，可以看出总共有35315条数据
data.info()


# In[5]:


# 统计评论文本长度信息,从平均长度可以看出属于短文本
data['text_a'].map(len).describe()


# In[6]:


# 统计数据集中7种情感类别标签的分布情况
data['label'].value_counts()


# In[8]:


# 可视化标签分布情况
get_ipython().run_line_magic('matplotlib', 'inline')
data['label'].value_counts(normalize=True).plot(kind='bar');


# ## 3.2 数据清洗

# In[9]:


# 导入所需包
import re
import os
import shutil
from tqdm import tqdm
from collections import defaultdict

# 定义数据清洗函数:

# 清洗分隔字符
def clean_duplication(text):
    left_square_brackets_pat = re.compile(r'\[+')
    right_square_brackets_pat = re.compile(r'\]+')
    punct = [',', '\\.', '\\!', '，', '。', '！', '、', '\?', '？']

    def replace(string, char):
        pattern = char + '{2,}'
        if char.startswith('\\'):
            char = char[1:]
        string = re.sub(pattern, char, string)
        return string

    text = left_square_brackets_pat.sub('', text)
    text = right_square_brackets_pat.sub('', text)
    for p in punct:
        text = replace(text, p)
    return text

def emoji2zh(text, inverse_emoji_dict):
    for emoji, ch in inverse_emoji_dict.items():
        text = text.replace(emoji, ch)
    return text

# 清洗数据集中特殊表情，通过json文件的映射用中文替代表情
def clean_emotion(data_path, emoji2zh_data, save_dir, train=True):
    data = defaultdict(list)
    filename = os.path.basename(data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        for line in tqdm(texts, desc=data_path):
            if train:
                id_, text, label = line.strip().split('\t')
            else:
                id_, text = line.strip().split('\t')
            data['id'].append(id_)
            text = emoji2zh(text, emoji2zh_data)
            text = clean_duplication(text)
            data['text_a'].append(text)
            if train:
                data['label'].append(label)
    df = pd.DataFrame(data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, filename), index=False,
              encoding='utf8', header=False, sep='\t')
    return df


# In[10]:


# 读取表情映射json文件（放在work目录下，文件名为emoji2zh.json），用于替换表情为中文字符
import json
emoji2zh_data = json.load(open('/home/aistudio/work/emoji2zh.json', 'r', encoding='utf8'))


# In[11]:


# 对数据进行数据清洗
data = clean_emotion('/home/aistudio/data/data100731/OCEMOTION.csv',emoji2zh_data,'./')


# In[12]:


# 去掉无用的id列，保存其格式为text_a,label
data = data[['text_a', 'label']]


# ## 3.3 转换情感类别标签
# 
# 由于类别名为英文，此处主要将英文类别名转为中文类别名，从而更好应用于中文情感分析系统中去！

# In[13]:


# 替换数据集中标签,{'sadness': '难过', 'happiness': '愉快', 'like': '喜欢', 'anger': '愤怒', 'fear': '害怕', 'surprise': '惊讶', 'disgust': '厌恶'}
data.loc[data['label']=='sadness', 'label'] = '难过'
data.loc[data['label']=='happiness', 'label'] = '愉快'
data.loc[data['label']=='like', 'label'] = '喜欢'
data.loc[data['label']=='anger', 'label'] = '愤怒'
data.loc[data['label']=='fear', 'label'] = '害怕'
data.loc[data['label']=='surprise', 'label'] = '惊讶'
data.loc[data['label']=='disgust', 'label'] = '厌恶'


# **下面提供了两种较常见的数据集划分方式，可以根据具体需要或效果进行选择：**

# In[14]:


# # # 划分方式1：根据比例直接划分训练、验证和测试集
# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(data, test_size=0.2)
# train_data,valid_data=train_test_split(train_data, test_size=0.2)

# # 对数据进行随机打乱
# from sklearn.utils import shuffle
# train_data = shuffle(train_data)
# valid_data = shuffle(valid_data)
# test_data = shuffle(test_data)

# # 保存划分好的数据集文件
# train_data.to_csv('./train.csv', index=False, sep="\t") # 训练集
# valid_data.to_csv('./valid.csv', index=False, sep="\t")  # 验证集
# test_data.to_csv('./test.csv', index=False, sep="\t")   # 测试集

# print('训练集长度：', len(train_dat), '验证集长度：', len(valid_data), '测试集长度', len(test_data))


# In[15]:


# 划分方式2：根据具体类别按8：1：1去划分训练、验证和测试集,这样可以使得数据尽量同分布

from sklearn.utils import shuffle
train = pd.DataFrame()  # 训练集
valid = pd.DataFrame()  # 验证集
test = pd.DataFrame()  # 测试集

tags = data['label'].unique().tolist()  # 按照该标签进行等比例抽取

# 根据数据集的类别按8:1:1的比例划分训练、验证和测试集并随机打乱后保存
for tag in tags:
    # 随机选取0.2的数据作为训练和验证集
    target = data[(data['label'] == tag)]
    sample = target.sample(int(0.2 * len(target)))
    sample_index = sample.index
    # 将剩余0.8的数据作为训练集
    all_index = target.index
    residue_index = all_index.difference(sample_index)  # 去除sample之后剩余的数据
    residue = target.loc[residue_index]
    # 对划分出来的0.2的数据集按等比例进行测试集和验证集的划分
    test_sample = sample.sample(int(0.5 * len(sample)))
    test_sample_index = test_sample.index
    valid_sample_index = sample_index.difference(test_sample_index)
    valid_sample = sample.loc[valid_sample_index]
    # 拼接各个类别
    test = pd.concat([test, test_sample], ignore_index=True)
    valid = pd.concat([valid, valid_sample], ignore_index=True)
    train = pd.concat([train, residue], ignore_index=True)
    # 对数据进行随机打乱
    train = shuffle(train)
    valid = shuffle(valid)
    test = shuffle(test)

# 保存为tab分隔的文本
train.to_csv('train.csv', sep='\t', index=False)  # 训练集
valid.to_csv('valid.csv', sep='\t', index=False)  # 验证集
test.to_csv('test.csv', sep='\t', index=False)    # 测试集

print('训练集长度：', len(train), '验证集长度：', len(valid), '测试集长度', len(test))


# ![](https://ai-studio-static-online.cdn.bcebos.com/0ad5564287fb48b0b5754f741c25919b83fd8a30af2644e1a3979e9e77c336e0)

# ## 4.1 前置环境准备

# In[16]:


# 下载最新版本的paddlehub
get_ipython().system('pip install -U paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[17]:


# 导入paddlehub和paddle包
import paddlehub as hub
import paddle


# ## 4.2 加载预训练模型-ERNIE Tiny

# ERNIE Tiny 主要通过模型结构压缩和模型蒸馏的方法，将 ERNIE 2.0 Base 模型进行压缩。特点和优势如下：
# 
# a.采用 3 层 transformer 结构，线性提速 4 倍;
# 
# b.模型加宽隐层参数，从 ERNIE 2.0 的 768 扩展到 1024；
# 
# c.缩短输入文本的序列长度，降低计算复杂度，模型首次采用中文 subword 粒度输入，长度平均缩短 40%；
# 
# d.ERNIE Tiny 在训练中扮演学生角色，利用模型蒸馏的方式在 Transformer 层和 Prediction 层学习教师模型 ERNIE 2.0 模型对应层的分布和输出;
# 
# 综合优化能带来4.3倍的预测提速，具有更高的工业落地能力。

# ![](https://ai-studio-static-online.cdn.bcebos.com/39e3b9125e124af290dee3c00dcc7f871de727a04f6142f1a380b98f45be2aa8)

# In[18]:


# 设置要求进行分类的7个情感类别
label_list=list(data.label.unique())
print(label_list)

label_map = { 
    idx: label_text for idx, label_text in enumerate(label_list)
}
print(label_map)


# In[19]:


# 只需指定想要使用的模型名称和文本分类的类别数即可完成Fine-tune网络定义，在预训练模型后拼接上一个全连接网络（Full Connected）进行分类
# 此处选择ernie_tiny预训练模型并设置微调任务为7分类任务
model = hub.Module(name="ernie_tiny", task='seq-cls', num_classes=7, label_map=label_map)


# `hub.Module`的参数用法如下：
# 
# * `name`：模型名称，可以选择`ernie`，`ernie_tiny`，`bert-base-cased`， `bert-base-chinese`, `roberta-wwm-ext`，`roberta-wwm-ext-large`等。
# * `task`：fine-tune任务。此处为`seq-cls`，表示文本分类任务。
# * `num_classes`：表示当前文本分类任务的类别数，根据具体使用的数据集确定，默认为2，需要根据具体分类任务进行选定。

# ## 4.3 加载并处理数据

# In[20]:


# 导入依赖库
import os, io, csv
from paddlehub.datasets.base_nlp_dataset import InputExample, TextClassificationDataset


# In[21]:


# 数据集存放位置
DATA_DIR="/home/aistudio/data/data100731/"


# In[22]:


# 对数据进行处理，处理为模型可接受的格式
class OCEMOTION(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'train.csv'  # 训练集
        elif mode == 'test':
            data_file = 'test.csv'   # 测试集
        else:
            data_file = 'valid.csv'  # 验证集
        
        super(OCEMOTION, self).__init__(
            base_path=DATA_DIR,
            data_file=data_file,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            is_file_with_header=True,
            label_list=label_list
            )

    # 解析文本文件里的样本
    def _read_file(self, input_file, is_file_with_header: bool = False):
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t")
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    try:
                        example = InputExample(guid=seq_id, text_a=line[0], label=line[1])
                        seq_id += 1
                        examples.append(example)
                    except:
                        continue
                return examples
                
train_dataset = OCEMOTION(model.get_tokenizer(), mode='train', max_seq_len=128)  # max_seq_len根据具体文本长度进行确定，但需注意max_seq_len最长不超过512
dev_dataset = OCEMOTION(model.get_tokenizer(), mode='dev', max_seq_len=128)
test_dataset = OCEMOTION(model.get_tokenizer(), mode='test', max_seq_len=128)

# 查看训练集前3条
for e in train_dataset.examples[:3]:
    print(e)
# 查看验证集前3条
for e in dev_dataset.examples[:3]:
    print(e)
# 查看测试集前3条
for e in test_dataset.examples[:3]:
    print(e)


# ## 4.4 选择优化策略和运行配置

# In[23]:


# 优化器的选择，此处使用了AdamW优化器
optimizer = paddle.optimizer.AdamW(learning_rate=4e-5, parameters=model.parameters())


# In[24]:


# 运行配置
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt', use_gpu=True, use_vdl=True)      # fine-tune任务的执行者


# #### 运行配置
# 
# `Trainer` 主要控制Fine-tune任务的训练，是任务的发起者，包含以下可控制的参数:
# 
# * `model`: 被优化模型；
# * `optimizer`: 优化器选择；
# * `use_gpu`: 是否使用gpu训练；
# * `use_vdl`: 是否使用vdl可视化训练过程；
# * `checkpoint_dir`: 保存模型参数的地址；
# * `compare_metrics`: 保存最优模型的衡量指标；

# ## 4.5 模型训练和验证
# 
# 注意模型的训练需要GPU环境，在模型训练时可以通过下方的'性能监控'或者在终端输入'nvdia-smi'命令查看显存占用情况，若显存不足可以适当调小batch_size

# In[25]:


trainer.train(train_dataset, epochs=5, batch_size=256, eval_dataset=dev_dataset, save_interval=1)   # 配置训练参数，启动训练，并指定验证集。


# `trainer.train` 主要控制具体的训练过程，包含以下可控制的参数：
# 
# * `train_dataset`: 训练时所用的数据集；
# * `epochs`: 训练轮数；
# * `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
# * `num_workers`: works的数量，默认为0；
# * `eval_dataset`: 验证集；
# * `log_interval`: 打印日志的间隔， 单位为执行批训练的次数。
# * `save_interval`: 保存模型的间隔频次，单位为执行训练的轮数。

# ## 4.6 在测试集上评估当前训练模型

# In[26]:


# 在测试集上评估当前训练模型
result = trainer.evaluate(test_dataset, batch_size=128) 


# In[27]:


# 进阶扩展： 使用F1-score指标对测试集上效果进行更官方的评测
import numpy as np
# 读取测试集文件
df = pd.read_csv('./test.csv',sep = '\t')

news1 = pd.DataFrame(columns=['label'])
news1['label'] = df["label"]
news = pd.DataFrame(columns=['text_a'])
news['text_a'] = df["text_a"]

# 首先将pandas读取的数据转化为array
data_array = np.array(news)
# 然后转化为list形式
data_list =data_array.tolist()

# 对测试集进行预测得到预测的类别标签
y_pre = model.predict(data_list, max_seq_len=128, batch_size=128, use_gpu=True)

# 测试集的真实类别标签
data_array1 = np.array(news1)
y_val =data_array1.tolist()

# 计算预测结果的F1-score
from sklearn.metrics import precision_recall_fscore_support,f1_score,precision_score,recall_score
f1 = f1_score(y_val, y_pre, average='macro')
p = precision_score(y_val, y_pre, average='macro')
r = recall_score(y_val, y_pre, average='macro')
print(f1, p, r)


# ## 4.7 模型预测

# In[29]:


# 要进行预测的数据
data = [
    # 难过
    ["你也不用说对不起,只是若相惜"],
    # 愉快
    ["幸福其实很简单"],
    # 害怕
    ["恐惧感啊。生病"],
    # 喜欢
    ["待你长发及腰,我们一起耕耘时光。我愿等待"]
]

# 定义要进行情感分类的7个类别
label_list=['难过', '愉快', 
'喜欢', '愤怒', '害怕'
, '惊讶', '厌恶']
label_map = {
    idx: label_text for idx, 
    label_text in enumerate(label_list)
}

# 加载训练好的模型
model = hub.Module(
    name='ernie_tiny',
    task='seq-cls',
    num_classes=7,
    load_checkpoint='./ckpt/best_model/model.pdparams',
    label_map=label_map)

# 进行模型预测
results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))


# In[26]:


# 将刚才训练好的最优模型参赛移动到work目录下，从而更好保存！
get_ipython().system('cp -r /home/aistudio/data/data100731/ckpt/best_model/model.pdparams /home/aistudio/work/中文微情感分析系统/best_model/')


# **可视化界面演示：**
# 
# 1. 单条文本情感分析页面：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/5bd476711b7e4bc5bf43185d917efa7deb571f11d6e54c1d8c76d3d09cc35210)
# 
# 2. 批量文本情感分析页面：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/08a11931b4834349aa3eef1e356a20314de1ebc25faf4edf9a204d31a6fe1fec)
