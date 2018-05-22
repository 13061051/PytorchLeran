
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np;
import os
import jieba
import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_legal(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True


def extract_chinese(line):
    res = ""
    for word in line:
        if is_legal(word):
            res = res + word
    return res;
def words2line(words):
    line = ""
    for word in words:
        line = line + " " + word
    return line
#数据预处理函数，在dir文件夹下每个子文件是一类内容
def datahelper(dir):
#返回为文本，文本对应标签
    labels_index={}
    index_lables={}
    num_recs=0
    fs = os.listdir(dir)
    MAX_SEQUENCE_LENGTH = 200
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 20
    VALIDATION_SPLIT = 0.2
    i = 0;
    for f in fs:
        labels_index[f] = i;
        index_lables[i] = f
        i = i + 1;
    print(labels_index)
    texts = []
    labels = []  # list of label ids
    for la in labels_index.keys():
        print(la + " " + index_lables[labels_index[la]])
        la_dir = dir + "/" + la;
        fs = os.listdir(la_dir)
        for f in fs:
            file = open(la_dir + "/" + f, encoding='utf-8')
            lines = file.readlines();
            text = ''
            for line in lines:
                if len(line) > 5:
                    line = extract_chinese(line)
                    words = jieba.lcut(line, cut_all=False, HMM=True)
                    text = words
                    texts.append(text)
                    labels.append(labels_index[la])
                    num_recs = num_recs + 1
    return texts,labels,labels_index,index_lables
#load word 2 vetc，加载词向量，可以事先预训练
def getw2v():
    model_file_name = 'new_model_big.txt'
    # 模型训练，生成词向量
    '''
    sentences = w2v.LineSentence('trainword.txt')
    model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
    model.save(model_file_name)
    '''
    model = w2v.Word2Vec.load(model_file_name)
    return model;

train_dir = 'new_data'

texts,labels,labels_index,index_lables=datahelper(train_dir)

#textCNN模型
class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        vocb_size = args['vocb_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix=args['embedding_matrix']
        #需要将事先训练好的词向量载入
        self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2),

                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,64,64)
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2)
                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.embeding(x)
        x=x.view(x.size(0),1,max_len,word_dim)
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        #print(x.size())
        output = self.out(x)
        return output
#词表
word_vocb=[]
word_vocb.append('')
for text in texts:
    for word in text:
        word_vocb.append(word)
word_vocb=set(word_vocb)
vocb_size=len(word_vocb)
#设置词表大小
nb_words=40000
max_len=64;
word_dim=20;
n_class=len(index_lables)

args={}
if nb_words<vocb_size:
    nb_words=vocb_size;
#textCNN调用的参数
args['vocb_size']=nb_words
args['max_len']=max_len
args['n_class']=n_class
args['dim']=word_dim

EPOCH=10;

texts_with_id=np.zeros([len(texts),max_len])
#词表与索引的map
word_to_idx={word:i for i,word in enumerate(word_vocb)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}
#每个单词的对应的词向量
embeddings_index = getw2v()
#预先处理好的词向量
embedding_matrix = np.zeros((nb_words, word_dim))
for word, i in word_to_idx.items():
    if i >= nb_words:
        continue
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
args['embedding_matrix']=torch.Tensor(embedding_matrix)
#构建textCNN模型
cnn=textCNN(args)

#生成训练数据，需要将训练数据的Word转换为word的索引
for i in range(0,len(texts)):
    if len(texts[i])<max_len:
        for j in range(0,len(texts[i])):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]
        for j in range(len(texts[i]),max_len):
            texts_with_id[i][j] = word_to_idx['']
    else:
        for j in range(0,max_len):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]

LR = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#损失函数
loss_function = nn.CrossEntropyLoss()
#训练批次大小
epoch_size=1000;
texts_len=len(texts_with_id)
print(texts_len)
#划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(texts_with_id, labels, test_size=0.2, random_state=42)


test_x=torch.LongTensor(x_test)
test_y=torch.LongTensor(y_test)
train_x=x_train
train_y=y_train

test_epoch_size=300;
for epoch in range(EPOCH):

    for i in range(0,(int)(len(train_x)/epoch_size)):

        b_x = Variable(torch.LongTensor(train_x[i*epoch_size:i*epoch_size+epoch_size]))

        b_y = Variable(torch.LongTensor((train_y[i*epoch_size:i*epoch_size+epoch_size])))
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(str(i))
        print(loss)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (b_y == pred_y)
        acc = acc.numpy().sum()
        accuracy = acc / (b_y.size(0))

    acc_all = 0;
    for j in range(0, (int)(len(test_x) / test_epoch_size)):
        b_x = Variable(torch.LongTensor(test_x[j * test_epoch_size:j * test_epoch_size + test_epoch_size]))
        b_y = Variable(torch.LongTensor((test_y[j * test_epoch_size:j * test_epoch_size + test_epoch_size])))
        test_output = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # print(pred_y)
        # print(test_y)
        acc = (pred_y == b_y)
        acc = acc.numpy().sum()
        print("acc " + str(acc / b_y.size(0)))
        acc_all = acc_all + acc

    accuracy = acc_all / (test_y.size(0))
    print("epoch " + str(epoch) + " step " + str(i) + " " + "acc " + str(accuracy))
