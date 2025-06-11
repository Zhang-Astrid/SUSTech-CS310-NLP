import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        # 定义模型的层
        self.hidden_size = 100
        self.word_embedding = nn.Embedding(word_vocab_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size * 6, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # 使用softmax进行归一化，dim=-1表示最后一维
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # 通过词向量层将输入x转换为embedding
        x = self.word_embedding(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 通过隐藏层进行处理并应用ReLU激活函数
        x = self.fc1(x)
        x = self.relu(x)
        # 最后通过输出层计算得分
        x = self.fc2(x)
        x = self.relu2(x)
        # 使用Softmax激活函数进行归一化
        x = self.fc3(x)
        x = self.softmax(x)
        ### END YOUR CODE ###
        return x


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        self.hidden_size = 100  # 增加隐藏层大小
        self.word_embedding = nn.Embedding(word_vocab_size, self.hidden_size)
        self.pos_embedding = nn.Embedding(pos_vocab_size, self.hidden_size)
        # # 增加 Dropout 层防止过拟合
        # self.dropout = nn.Dropout(p=0.5)  # Dropout 比例为 0.5，您可以根据需要调整
        self.fc1 = nn.Linear(12 * self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # 使用softmax进行归一化，dim=-1表示最后一维

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != 12:
            raise ValueError(f"Expected input shape [batch_size, 12], got {x.shape}")

        batch_size = x.size(0)
        word_ids = x[:, :6].long()  # First 6 columns for word indices
        pos_ids = x[:, 6:].long()  # Last 6 columns for POS tag indices
        if torch.any(word_ids < 0) or torch.any(pos_ids < 0):
            raise ValueError("Found negative IDs in input")

        # Embedding for words and POS tags
        word_embedded = self.word_embedding(word_ids)
        pos_embedded = self.pos_embedding(pos_ids)

        word_embedded = word_embedded.view(batch_size, 6, -1)
        pos_embedded = pos_embedded.view(batch_size, 6, -1)

        # features = torch.cat([word_embedded, pos_embedded], dim=2)# Wrong here
        features = torch.cat([word_embedded, pos_embedded], dim=1)
        features = features.view(batch_size, -1)

        hidden1 = self.fc1(features)
        hidden1 = self.relu(hidden1)
        hidden2 = self.fc2(hidden1)
        hidden2 = self.relu(hidden2)
        hidden3 = self.fc3(hidden2)
        output = self.softmax(hidden3)
        return output


if __name__ == "__main__":
    word_vocab_size = 1000  # 假设词汇表大小为1000
    pos_vocab_size = 50  # 假设词性词汇表大小为50
    output_size = 10  # 假设输出类别数为10
    batch_size = 100  # 假设batch size为4

    # 创建模型实例
    # model = BaseModel(word_vocab_size, output_size)
    model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    input_data = torch.zeros((batch_size, 12), dtype=torch.long)
    output_base = model(input_data)
    print(f"Output shape: {output_base.shape}")
