from GPT import *
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import math
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset




with open("C:\\Code\\Jupter\\ML\\Project\\GPT_From_scratch\\hamlet.txt", "r", encoding="utf-8") as f:
    text = f.read()   # read entire file as string

#convert text to list of words and remove punctuation and single character words
words = word_tokenize(text)
word=[w for w in words if w not in ['.',',','!','?',';',':','(',')','[',']','{','}','"',"'",'-'] and w.isalpha() and len(w)>1]

#create sequences of words for training
tokenizer = Tokenizer()
tokenizer.fit_on_texts(word)
sequences = tokenizer.texts_to_sequences(word)
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)


#split data into input and target sequences
input=[]
for i in text.split('\n'):
    load=tokenizer.texts_to_sequences([i])[0]
    
    for j in range(1, len(load)):
        input.append(load[:j+1])

# Create Pad sequences 

max_len = max(len(seq) for seq in input)

padded = torch.zeros(len(input), max_len, dtype=torch.long)

for i, seq in enumerate(input):
  
    padded[i, -len(seq):] = torch.tensor(seq)


#x is input sequence and y is target word (last word of the sequence)
x,y =padded[:,:-1], padded[:,-1]

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)   

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#Testing data preparation
x_test_tensor = torch.tensor(x_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class transformer(nn.Module):

    def __init__(self, d_model, vocab_size):

        super(transformer, self).__init__()
        
    

        self.embedding = Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.attention = multihead_attention(d_model, num_heads=2)

        self.norm1 = layernormalition(d_model)

        self.feedforward = feedforward(d_model, d_ff=48)

        self.norm2 = layernormalition(d_model)

        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x):

        output = self.embedding(x)

        output = self.pos_encoding(output)
        res1=output 
        output = self.attention(output)

        # Add & Norm
        output = output + res1 
        output = self.norm1(output)

        res2=output 
        output = self.feedforward(output)
        
        # Add & Norm
        output = output + res2
        output = self.norm2(output)

        output = self.linear(output)

        return output
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

criterion = nn.CrossEntropyLoss()

model = transformer(d_model=6, vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epochs in range(10000):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        output = output[:, -1, :]   # take last token prediction
        loss = criterion(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        preds = torch.argmax(output, dim=-1)

        running_loss += loss.item()
        
    
    print(f"Epoch [{epochs+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ==========================
# Evaluation
# ==========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        predicted = outputs[:, -1, :] 
        predicted = torch.argmax(predicted, dim=-1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
