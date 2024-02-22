import torch
from torch import nn
from torch.nn import functional as F

class ConvBNReLU(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        res = self.block(x)
        res += x
        return F.relu(res)

class CNN(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            # Input shape: batch_size, 1, height, width
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256 * (image_height // 16) * (image_width // 16), out_features=num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out


class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
    def forward(self, x):
        outputs, _ = self.bigru(x)
        return outputs

class LinearCRF(nn.Module):
    def __init__(self, input_dim, tagset_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, tagset_size)
    def forward(self, feats):
        logits = self.linear(feats)
        trans = torch.zeros((logits.shape[0], logits.shape[1], logits.shape[1])).to(device)
        for i in range(trans.size(1)):
            if i == 0:
                trans[:,i,i+1] = -1e9
            elif i < trans.size(1)-1:
                trans[:,i,i-1] = -1e9
                trans[:,i,i+1] = -1e9
            else:
                trans[:,i,i-1] = -1e9
        return logits, trans

class CRNN(nn.Module):
    def __init__(self, num_classes, cnn_config={}, rnn_config={}):
        super().__init__()
        self.cnn = CNN(**cnn_config).to(device)
        self.rnn = BiGRU(input_dim=cnn_config['output_dim'], **rnn_config)
        self.crf = LinearCRF(input_dim=rnn_config['hidden_dim']*2, tagset_size=num_classes)
    def forward(self, x):
        cnn_outputs = self.cnn(x)
        rnn_inputs = cnn_outputs.unsqueeze(-1).permute(0,2,1)
        rnn_outputs, _ = self.rnn(rnn_inputs)
        crf_inputs = rnn_outputs
        logits, trans = self.crf(crf_inputs)
        return logits, trans




# def train_epoch(model, iterator, optimizer, criterion):
#     epoch_loss = 0
#     model.train()
#     for batch in iterator:
#         texts, targets = batch
#         optimizer.zero_grad()
#         logits, trans = model(texts)
#         loss = criterion(logits, targets, trans)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / len(iterator)
# def evaluate(model, iterator, criterion):
#     epoch_loss = 0
#     model.eval()
#     with torch.no_grad():
#         for batch in iterator:
#             texts, targets = batch
#             logits, trans = model(texts)
#             loss = criterion(logits, targets, trans)
#             epoch_loss += loss.item()
#     return epoch_loss / len(iterator)


# # Instantiate the model, criterion, and optimizer here
# model = CRNN(num_classes=len(unique_chars)+1, cnn_config={'input_channels': 1, 'output_dim': 256}, rnn_config={'input_dim': 256, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.2})
# criterion = CTCLoss()
# optimizer = AdamW(model.parameters(), lr=0.001)
# # Load pretrained weights here
# pretrained_state_dict = torch.load('path/to/pretrained/weights.pt')
# model.load_state_dict(pretrained_state_dict)
# # Train the model here
# for epoch in range(num_epochs):
#     train_loss = train_epoch(model, train_iter, optimizer, criterion)
#     val_loss = evaluate(model, valid_iter, criterion)
#     print("Epoch {} | Training Loss: {:.4f} | Validation Loss: {:.4f}".format(epoch+1, train_loss, val_loss))