import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.nn.functional as F
import pdb

import numpy as np



seqs = []
refs = []
scores = []
for f in range(1, 4):
    scores1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl-random-chebi-{f}-epochs300-loop.mistral-7B.scores.npy")
    embeds1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl-random-chebi-{f}-epochs300-embeds.npz")
    seqs1 = torch.cat([torch.zeros(3297, 5-f, 64), torch.tensor(embeds1['embeds']).flip(dims=[1])], dim=1)
    refs1 = np.reshape(embeds1['test_pool'], (embeds1['test_pool'].shape[0],1,-1))
    seqs.append(seqs1)
    refs.append(refs1)
    scores.append(scores1)


seqs = torch.cat(seqs)
scores = torch.tensor(np.concatenate(scores))
refs = torch.tensor(np.concatenate(refs))
print (seqs.shape, refs.shape)

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMWithAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size//2)
        #self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.attention_layer = nn.Linear(hidden_size, 1)  # To compute attention weights
        self.dropout = nn.Dropout(0.2)

    def forward(self, i, x):
        q = self.query(i)
        x_out, (h, _) = self.lstm(x)  
        h = self.dropout(h)
        #input_hidden = q.unsqueeze(1)
        attn_scores = torch.bmm(x_out, q.transpose(1, 2)) 
        attn_weights = F.softmax(attn_scores, dim=1)

        context_vector = torch.sum(attn_weights * x_out, dim=1) 

        combined = torch.cat((q.squeeze(1), h.squeeze(0)), dim=1) 

        z = F.relu(self.fc1(combined))  # First transformation with ReLU
        #z = F.relu(self.fc2(z))
        score = self.fc3(z)
        return score

batch_size = 32

dataset = TensorDataset(refs, seqs, scores)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


input_size = 64
hidden_size = 64
num_layers = 2
epochs = 100
batch_size = 16


model = LSTMWithAttention(input_size, hidden_size)

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        ref, seq, score = batch
        outputs = model(ref, seq) 
        loss = criterion(outputs.squeeze(1), score.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
        
    # Validation phase
    model.eval()
    with torch.no_grad():  # No gradients needed for validation
        val_loss = 0.0
        for batch in val_loader:
            ref, seq, score = batch
            
            outputs = model(ref, seq)  # Outputs shape: [batch_size, output_size]
            loss = criterion(outputs, score)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


output = model(refs, seqs)
output = output.squeeze(1) #topk(10)
output.shape
output.topk(100)




