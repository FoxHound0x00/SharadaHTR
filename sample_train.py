from data.dataset import SharadaDataset
from dataloader import SharadaDataLoader
from model import SharadaCRNN
from transforms import Rescale, Deskew, toRGB, ToTensor, Normalise
from train import Train

from torchvision.transforms import Compose
from torchvision import transforms
import torch

# use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Create the CTCDataset
dataset = SharadaDataset(txt_dir='/home/kunal/Desktop/Sharada_Jan/extracted_dir/',       
                        img_dir='/home/kunal/Desktop/Sharada_Jan/extracted_dir/',                        
                        transform=Compose([                                                     
                            # Deslant(),                                                           
                            PadResize(output_size=(64,200)),
                            ToTensor(), # converted to Tensor
                            Normalize_Cust(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])) 


# Create a dataloader
dl = SharadaDataLoader(dataset,                                                       
                       batch_size=(120,240),                                          
                       validation_split=0.2,                                          
                       shuffle=True,                                                  
                       seed=3407,                                                     
                       device=device)                                                    

# Create the model
model = SharadaHTR(chan_in=3,                                                            
                 time_step=150,                                                                                                          
                 feature_size=512,                                                      
                 hidden_size=512,                        
                 output_size=len(dataset.char_dict) + 1,                                
                 num_rnn_layers=4,                                                     
                 rnn_dropout=0)
# model.load_pretrained_resnet()                                                         
model.to(device)                                                                           


learn = Train(model=model,                                                           
                dataloader=dl,                                                         
               decode_map={v:k for k,v in dataset.char_dict.items()})

# learn.freeze()                                                                         # freeze & unfreeze the conv weights
# log, lr = learn.find_lr(start_lr=1e-5, end_lr=1e1, wd=0.1)                                                                                      
# based on https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

learn.fit_one_cycle(epochs=10, max_lr=1e-3, base_lr=1e-4, wd=0.1)