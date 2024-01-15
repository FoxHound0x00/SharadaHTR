import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from skimage.color import rgb2gray
from skimage.transform import rotate
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from torchvision import transforms
from torchinfo import summary

from utils import *
from dataset import SharadaDataset
from dataloader import SharadaDataLoader
from transforms import PadResize, Deskew, toRGB, ToTensor, Normalize_Cust

os.makedirs("chk_pts/", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



dataset = SharadaDataset(txt_dir='data/dataset/',       
                        img_dir='data/dataset/',                        
                        transform=Compose([                                                     
                            # Deslant(),                                                           
                            PadResize(output_size=(64,200)),
                            ToTensor(), # converted to Tensor
                            Normalize_Cust(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])) 

dl = SharadaDataLoader(dataset,                                                       
                       batch_size=(120,240),                                          
                       validation_split=0.2,                                          
                       shuffle=True,                                                  
                       seed=3407,                                                     
                       device=str(device))     
                                               
crnn_model = CRNN(input_channels=3, hidden_size=256, num_layers=2, num_classes=len(dataset.char_dict) + 1).to(device)
optimizer = Adam(crnn_model.parameters(), lr=0.001)

# print(crnn_model)
# summary(crnn_model, (100, 3, 64, 64))

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
# loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)

train_loader, val_loader = dl()

writer = SummaryWriter()
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    crnn_model.train()  
    total_loss = 0.0

    for images, targets, lengths in train_loader:  
        # print("Here:",images)
        images = images.to(device)
        targets = targets.to(device)

        logits = crnn_model(images)
        print(logits.shape)

        # Calculate the CTC loss
        loss = ctc_loss(logits, targets, lengths, lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Validation
    if (epoch + 1) % 1 == 0:  # You can adjust the frequency of validation
        crnn_model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for val_images, val_targets, val_lengths in val_loader:  # Assuming dl() returns validation_loader
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)

                val_logits = crnn_model(val_images)
                val_loss += ctc_loss(val_logits, val_targets, val_lengths, val_lengths).item()

                _, predicted_labels = torch.max(val_logits, 2)
                predicted_labels = ["".join([dataset.char_list[c] for c in row if c != 0]) for row in predicted_labels.cpu().numpy()]

                for pred, target in zip(predicted_labels, val_targets.cpu().numpy()):
                    distance = levenshtein_distance(pred, "".join([dataset.char_list[c] for c in target if c != 0]))

                    writer.add_scalar('LevenshteinDistance/Validation', distance, epoch)

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        crnn_model.train()

        print(f'Validation Loss: {avg_val_loss:.4f}')

torch.save(crnn_model.state_dict(), 'chk_pts/crnn_model.pth')
writer.close()