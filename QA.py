train = pd.read_json('train.jsonl', lines=True, orient='records')

print('True: {}%'
    .format((train['answer']
    .value_counts()[1]/train.shape[0]*100)
    .round(2)))train['question'].str.split().apply(len).value_counts().plot.bar()

from math import *

avg_question_words = reduce(lambda x,y: x + y,
    list(map(lambda x: len(x.split()),train['question'].to_list())))/train.shape[0]
avg_question_symbols = reduce(lambda x,y: x + y,
    list(map(lambda x: len(x),train['question'].to_list())))/train.shape[0]

avg_passage_words = reduce(lambda x,y: x + y,
    list(map(lambda x: len(x.split()),train['passage'].to_list())))/train.shape[0]
avg_passage_symbols = reduce(lambda x,y: x + y,
    list(map(lambda x: len(x),train['passage'].to_list())))/train.shape[0]


print('''
    Q_avg_words: {}
    Q_avg_symbols: {}
    P_avg_words: {}
    P_avg_symbols: {}
    '''.format(round(avg_question_words,2),
        round(avg_question_symbols,2),
        round(avg_passage_words,2),
        round(avg_passage_symbols,2)))


max_question_words = reduce(lambda x,y: max(x,y),
    list(map(lambda x: len(x.split()),train['question'].to_list())))
max_question_symbols = reduce(lambda x,y: max(x,y),
    list(map(lambda x: len(x),train['question'].to_list())))

max_passage_words = reduce(lambda x,y: max(x,y),
    list(map(lambda x: len(x.split()),train['passage'].to_list())))
max_passage_symbols = reduce(lambda x,y: max(x,y),
    list(map(lambda x: len(x),train['passage'].to_list())))


print('''
    Q_max_words: {}
    Q_max_symbols: {}
    P_max_words: {}
    P_max_symbols: {}
    '''.format(round(max_question_words,2),
        round(max_question_symbols,2),
        round(max_passage_words,2),
        round(max_passage_symbols,2)))

train['question'].str.split().apply(len).value_counts().plot.bar()


from sklearn.metrics import accuracy_score
import fasttext


test = pd.read_json('dev.jsonl', lines=True, orient='records')



y = test['answer']
y_pred = pd.Series([True for i in range(test.shape[0])])


accuracy_score(y, y_pred)

fasttext_train = '__label__' + train['answer'].astype(int).astype(str) + ' ' + train['question'] + ' ' + train['passage']
fasttext_test = '__label__' + test['answer'].astype(int).astype(str) + ' ' + test['question'] + ' ' + test['passage']


np.savetxt('fasttext_train.train', fasttext_train.values, fmt='%s')
np.savetxt('fasttext_test.test', fasttext_test.values, fmt='%s')




model = fasttext.train_supervised(input='fasttext_train.train')



model.test('fasttext_test.test')[1]




from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset

import transformers

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, SchedulerCallback, F1ScoreCallback
from catalyst.utils import set_global_seed, prepare_cudnn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import re
import nltk
import string
from nltk.corpus import stopwords

def text_prepare(text):
    """
        text: a string
        
    """
    text = text.lower()
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub('[^A-Za-z\. ]', '', text)
    stopWords = set(stopwords.words('english'))
    for stopWord in stopWords:
        # text = text.replace(stopWord, '')
        text = re.sub('^{}$'.format(stopWord), '', text)
    return text



bert_train = pd.DataFrame()
bert_train['sentence'] = (train['question'] + ' ' + train['passage']).apply(text_prepare)
bert_train['label'] = train['answer'].astype(int)
bert_test_valid = pd.DataFrame()
bert_test_valid['sentence'] = (test['question'] + ' ' + test['passage']).apply(text_prepare)
bert_test_valid['label'] = test['answer'].astype(int)

bert_test, bert_valid = train_test_split(bert_test_valid, test_size=0.5)




model_name = 'bert-base-uncased'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# test_sentence = "Hide new secretions from the parental units."
# print(tokenizer.tokenize(test_sentence))
# print(tokenizer.encode(test_sentence))



class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        max_seq_length = 775
        
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_plus = [ 
            tokenizer.encode_plus(item, max_length=max_seq_length, pad_to_max_length=True) 
            for item in data['sentence']
        ]
        
        self.input_ids = torch.tensor([ i['input_ids'] for i in self.encoded_plus ], dtype=torch.long) 
        self.attention_mask = torch.tensor([ i['attention_mask'] for i in self.encoded_plus ], dtype=torch.long) 
        self.token_type_ids = torch.tensor([ i['token_type_ids'] for i in self.encoded_plus ], dtype=torch.long) 
        self.target = torch.tensor(list(self.data['label']), dtype=torch.long) 


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'targets': self.target[idx]
        }




class BertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels = num_labels
        )
        
        self.bert = transformers.BertModel.from_pretrained(
            pretrained_model_name, 
            num_labels=num_labels
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        
        assert attention_mask is not None, "attention mask is none"
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2

pretrained_model_name = 'google/bert_uncased_L-4_H-256_A-4'
tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name)
model = BertForSequenceClassification(pretrained_model_name, num_labels=num_labels)

model.to(device)
print("Success!")


batch_size = 8

train_dataset = TextClassificationDataset(bert_train, tokenizer)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

valid_dataset = TextClassificationDataset(bert_valid, tokenizer)
valid_sampler = RandomSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

test_dataset = TextClassificationDataset(bert_test, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

dataloaders = {
    "train": train_dataloader,
    "valid": valid_dataloader,
    "test": test_dataloader    
}

print(f"Dataset size: {len(train_dataloader)}")



seed = 404
set_global_seed(seed)
prepare_cudnn(True)

# Гиперпараметры для обучения модели. Подбери нужные для каждой модели.

epochs = 10
lr = 1e-5
warmup_steps = len(train_dataloader) // 2
t_total = len(train_dataloader) * epochs




optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()], "weight_decay": 0.0},
]

criterion = torch.nn.CrossEntropyLoss()
optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
)







log_dir = 'logs/'



torch.cuda.empty_cache()
torch.cuda.is_available()




train_val_loaders = {
    "train": train_dataloader,
    "valid": valid_dataloader
}

runner = SupervisedRunner(
    input_key=(
        "input_ids",
        "attention_mask",
        "token_type_ids"
    )
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=train_val_loaders,
    callbacks=[
        AccuracyCallback(num_classes=num_labels),
        SchedulerCallback(mode='batch'),
    ],
    logdir=log_dir,
    num_epochs=epochs,
    verbose=False,
)


logits = runner.predict_loader(model=model, loader=test_dataloader, resume=log_dir+'checkpoints/best.pth')
y_pred = []
counter = 0
softmax = torch.nn.Softmax(dim=1)
for i in logits:
    for j in softmax(i['logits']).argmax(axis=1).tolist():
        y_pred.append(j)

y_true = []

for i in test_dataloader:
    for j in i['targets'].tolist():
        y_true.append(j)
        

accuracy_score(y_true, y_pred)


