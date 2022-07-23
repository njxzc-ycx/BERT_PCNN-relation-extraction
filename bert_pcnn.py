import pandas as pd
import torch
from transformers import BertModel
from transformers import BertTokenizer
import os
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



def split_data(path):
    all_datas = []

    with open(os.path.join("data", path), 'r', encoding="utf-8") as f:
        for lines in f:
            data = lines.split()
            all_datas.append(data)

    # 划分训练集、验证集、测试集
    train_datasets, dev_test_datasets = train_test_split(all_datas, test_size=0.3, random_state=2)
    dev_datasets, test_datasets = train_test_split(dev_test_datasets, test_size=0.3, random_state=2)

    return train_datasets, dev_datasets, test_datasets

def pro_data(datasets, num, max_len):

    all_text = []
    all_label = []
    all_pos1 = []
    all_pos2 = []
    all_mask = []

    for data in tqdm(datasets[:num], desc="正在处理数据"):
        if data == "":
            continue

        position1 = []
        position2 = []
        mask = []

        e1 = data[0]
        e2 = data[1]
        label = data[2]
        text = data[3]

        all_text.append(text)
        all_label.append(label)

        pos1_index = min(text.index(e1), max_len)
        pos2_index = min(text.index(e2), max_len)

        for i in range(max_len):
            position1.append(min(i - pos1_index + max_len, 2 * max_len - 1))
            position2.append(min(i - pos2_index + max_len, 2 * max_len - 1))

        all_pos1.append(position1)
        all_pos2.append(position2)

        pos_min = min(pos1_index, pos2_index)
        pos_max = max(pos1_index, pos2_index)

        for i in range(max_len):
            if i <= pos_min - 1:
                mask.append(1)
            elif i <= pos_max - 1:
                mask.append(2)
            if len(text) < max_len:
                if i <= len(text) and i > pos_max - 1:
                    mask.append(3)
                elif i > len(text):
                    mask.append(0)
            else:
                if i <= len(text) and i > pos_max - 1:
                    mask.append(3)

        all_mask.append(mask)

    return all_text, all_label, all_pos1, all_pos2, all_mask


def build_label(train_label):
    label_2_index = {}
    for label in train_label:
        if label not in label_2_index:
            label_2_index[label] = len(label_2_index)
    return label_2_index, list(label_2_index)

class MyDataset(Dataset):
    def __init__(self, all_text, all_label, all_pos1, all_pos2, all_mask, label_2_index, tokenizer, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.all_pos1 = all_pos1
        self.all_pos2 = all_pos2
        self.all_mask = all_mask
        self.label_2_index = label_2_index
        self.max_length = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]
        pos1 = self.all_pos1[index]
        pos2 = self.all_pos2[index]
        mask = self.all_mask[index]

        text_index = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.max_length,  # max_length + 2 因为bert添加了左右起始结束字符
                                           padding="max_length", truncation=True, return_tensors="pt")

        label_index = self.label_2_index[label]
        label_index = torch.tensor(label_index)

        pos1 = torch.tensor(pos1)
        pos2 = torch.tensor(pos2)
        mask = torch.tensor(mask)

        return text_index.reshape(-1), label_index, pos1, pos2, mask

    def __len__(self):
        return self.all_text.__len__()


class PCNNModel(nn.Module):
    def __init__(self, embedding_dim, pos_dim, class_size, kernel_size, padding_size,
                 hidden_size, max_len, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.pos_dim = pos_dim
        self.input_dim = embedding_dim + 2 * pos_dim
        self.class_size = class_size
        self.max_len = max_len

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.bert = BertModel.from_pretrained(os.path.join(".", "bert_base_chinese"))
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        # self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.pos1_embeds = nn.Embedding(2 * max_len, self.pos_dim, padding_idx=0)
        self.pos2_embeds = nn.Embedding(2 * max_len, self.pos_dim, padding_idx=0)

        self.conv = nn.Conv1d(self.input_dim, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_len)
        masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(masks)
        self.mask_embedding.weight.requires_grad = False
        self._minus = 1e-6

        #         self.mask_embedding = nn.Embedding(4, 3)
        #         self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        #         self.mask_embedding.weight.requires_grad = False
        #         self._minus = -1e6

        self.hidden_size *= 3
        self.linear = nn.Linear(self.hidden_size, self.class_size)

        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, data_index, pos1, pos2, mask, label=None):
        bert_out = self.bert(data_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0：字符级别特征, bert_out1：篇章级别特征

        ps1_embeds = self.pos1_embeds(pos1)
        ps2_embeds = self.pos2_embeds(pos2)
        x = torch.cat([bert_out0,
                       ps1_embeds,
                       ps2_embeds], 2)

        # x = self.drop(x)
        x = x.transpose(1, 2)  # (B, EMBED, L)
        x = self.conv(x)  # (B, H, L)
        mask = 1 - self.mask_embedding(mask).transpose(1, 2)  # (B, L) -> (B, L, 3) -> (B, 3, L)
        pool1 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))  # (B, H, 1)
        pool2 = self.pool(self.act(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(self.act(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1).tanh()  # (B, 3H, 1)
        x = self.drop(x)
        x = x.squeeze(2)  # (B, 3H)
        pre = self.linear(x)

        if label is not None:
            loss = self.loss_fun(pre, label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)



if __name__ == "__main__":
    max_len = 50
    train_datasets, dev_datasets, test_datasets = split_data("train.txt")

    train_text, train_label, train_pos1, train_pos2, train_mask = pro_data(train_datasets, num=200000, max_len=max_len)
    dev_text, dev_label, dev_pos1, dev_pos2, dev_mask = pro_data(dev_datasets, num=20000, max_len=max_len)
    test_text, test_label, test_pos1, test_pos2, test_mask = pro_data(test_datasets, num=3000, max_len=max_len)

    label_2_index, index_2_label = build_label(train_label)

    tokenizer = BertTokenizer.from_pretrained(os.path.join(".", "bert_base_chinese"))

    batch_size = 12
    epoch = 100
    lr = 0.001

    kernel_size = 3
    padding_size = 1
    dropout = 0.3
    embedding_dim = 768
    pos_dim = 50
    hidden_size = 230
    class_size = len(label_2_index)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(train_text, train_label, train_pos1, train_pos2, train_mask, label_2_index, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = MyDataset(dev_text, dev_label, dev_pos1, dev_pos2, dev_mask, label_2_index, tokenizer, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(test_text, test_label, test_pos1, test_pos2, test_mask, label_2_index, tokenizer, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = PCNNModel(embedding_dim, pos_dim, class_size, kernel_size, padding_size, hidden_size, max_len, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_acc = -1  # 用于记录最优模型

    for e in range(epoch):
        model.train()
        for batch_text_index, batch_label_index, batch_pos1, batch_pos2, batch_mask in tqdm(train_dataloader):
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            batch_pos1 = batch_pos1.to(device)
            batch_pos2 = batch_pos2.to(device)
            batch_mask = batch_mask.to(device)

            loss = model(batch_text_index, batch_pos1, batch_pos2, batch_mask, batch_label_index)
            loss.backward()
            opt.step()
            opt.zero_grad()


        model.eval()
        right_num = 0
        for batch_text_index, batch_label_index, batch_pos1, batch_pos2, batch_mask in tqdm(dev_dataloader):
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            batch_pos1 = batch_pos1.to(device)
            batch_pos2 = batch_pos2.to(device)
            batch_mask = batch_mask.to(device)

            pre = model(batch_text_index, batch_pos1, batch_pos2, batch_mask)

            right_num += int(torch.sum(pre == batch_label_index))

            y_true = batch_label_index.reshape(-1).cpu().numpy()
            y_pre = pre.cpu().numpy()
            f1 = f1_score(y_true, y_pre, average="micro")

        acc = right_num / len(dev_text) * 100
        print(f"eopch:{e + 1}/{epoch}, loss:{loss:.5f}, dev_acc:{acc:.5f}, dev_f1:{f1:.5f}")
        if acc > best_acc:  # 此时为最优模型，所以保存下载
            print("保存模型中！")
            best_acc = acc
            torch.save(model.state_dict(), "bert_pcnn_best_model.pth")


    model.eval()
    right_num = 0
    true_label = []
    pre_label = []
    for batch_text_index, batch_label_index, batch_pos1, batch_pos2, batch_mask in tqdm(test_dataloader):
        batch_text_index = batch_text_index.to(device)
        batch_label_index = batch_label_index.to(device)
        batch_pos1 = batch_pos1.to(device)
        batch_pos2 = batch_pos2.to(device)
        batch_mask = batch_mask.to(device)

        pre = model(batch_text_index, batch_pos1, batch_pos2, batch_mask)

        right_num += int(torch.sum(pre == batch_label_index))

        y_true = batch_label_index.reshape(-1).cpu().numpy()
        y_pre = pre.cpu().numpy()

        for i in y_true:
            true_label.append(index_2_label[i])

        for i in y_pre:
            pre_label.append(index_2_label[i])


        f1 = f1_score(y_true, y_pre, average="micro")
    print(f"test--- :test_acc:{right_num / len(test_text) * 100:.5f}%, test_f1: {f1:.5f}")
    pd.DataFrame({"text": test_text, "label_true": true_label, "label_pre": pre_label}).to_csv("text_result2.csv", index=False)



