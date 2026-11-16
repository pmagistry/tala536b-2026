import torch
from torch.utils.data import DataLoader, Dataset, dataloader
import torch.nn as nn
import torch.optim as optim
import numpy as np


def tokenize(data):
    text = data['text']
    text = text.lower()
    text = text.split()
    data['text'] = text

    return data
"""
def vectorize(data, cats):

    map_cats = {}

    vocab = [token for d in data for token in tokenize(d['text'])]

    freqs =

    vec = np.zeros(len(vocab))

    for

    return vocab,
"""
def model_embeddings_pt(vocab_size, embed_size, hidden_size, output_size) :

    class RNN(nn.Module) :
        def __init__(self, vocab_size, embed_size, hidden_size, output_size) :
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, device='cuda')
            self.pool = nn.AvgPool1d(embed_size, stride=1)
            self.hidden = nn.RNN(embed_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.Softmax(output_size)

        def forward(self, x) :
            x = self.embedding(x)
            x = self.pool(x)
            init = torch.zeros(1, x.size(0), embed_size).to(x.device)
            out, _ = self.hidden(x, init)
            out = self.output(out[:,-1,:])
            return out


    model = RNN(vocab_size, embed_size, hidden_size, output_size)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    return model, optimizer, loss_function

def evaluate_model(model, data):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in data:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += 1 if predicted == labels else 0

    accuracy = 100 * correct / total

    return accuracy

def train_model(model, data, cats, n_epochs, optimizer, loss_function):

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        dataset = zip(data, cats)

        for inputs, labels in dataset:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('Epoch {}/{}'.format(epoch+1, n_epochs), 'Loss: {:.4f}'.format(epoch_loss / len(data)))

    return evaluate_model(model, data)
