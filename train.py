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

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
# loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)

train_loader, val_loader = dl()

writer = SummaryWriter()
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    crnn_model.train()  # Set the model to training mode
    total_loss = 0.0

    # Iterate over the training dataset
    for images, targets, lengths in train_loader:  # Assuming dl() returns train_loader
        # print("Here:",images)
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        logits, trans = crnn_model(images)

        # Calculate the CTC loss
        loss = ctc_loss(logits.permute(1, 0, 2), targets, lengths, lengths)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_loss = total_loss / len(train_loader)

    # Log the training loss to Tensorboard
    writer.add_scalar('Loss/Train', avg_loss, epoch)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Validation
    if (epoch + 1) % 1 == 0:  # You can adjust the frequency of validation
        crnn_model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        # Iterate over the validation dataset
        with torch.no_grad():
            for val_images, val_targets, val_lengths in val_loader:  # Assuming dl() returns validation_loader
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)

                # Forward pass
                val_logits, val_trans = crnn_model(val_images)

                # Calculate the CTC loss
                val_loss += ctc_loss(val_logits.permute(1, 0, 2), val_targets, val_lengths, val_lengths).item()

                _, predicted_labels = torch.max(val_logits, 2)
                predicted_labels = ["".join([dataset.char_list[c] for c in row if c != 0]) for row in predicted_labels.cpu().numpy()]

                for pred, target in zip(predicted_labels, val_targets.cpu().numpy()):
                    distance = levenshtein_distance(pred, "".join([dataset.char_list[c] for c in target if c != 0]))

                    writer.add_scalar('LevenshteinDistance/Validation', distance, epoch)

        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)

        # Log the validation loss to Tensorboard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f'Validation Loss: {avg_val_loss:.4f}')

# Save the trained model
torch.save(crnn_model.state_dict(), 'chk_pts/crnn_model.pth')

# Close Tensorboard writer
writer.close()