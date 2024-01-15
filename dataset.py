import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
class SharadaDataset(Dataset):
    """Scripture dataset Class."""

    def __init__(self, txt_dir, img_dir, transform=None, char_dict=None):
        """
        Args:
            txt_dir (string): Path to the txt file with labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.transform = transform
        self.max_len = 0
        self.char_list = " -ँंःअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅेैॉॊोौ्ॐ॒॑॓॔क़ख़ग़ज़ड़ढ़फ़य़ॠॢ।॥०१२३४५६७८९॰ॱॲॻॼॽॾ≈–|"
        if self.char_list is not None:
            chars = sorted(list(set(self.char_list)))
            self.char_dict = {c:i for i,c in enumerate(chars,1)}

        txt_files = os.listdir(self.txt_dir)
        self.txt_paths = [txt_file for txt_file in txt_files if txt_file.endswith('.txt')]
        img_files = os.listdir(self.img_dir)
        self.img_paths = [img_file for img_file in img_files if img_file.endswith('.jpg')]


    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img_filepath = os.path.join(self.img_dir,img_name)
        try:
            image = Image.open(img_filepath)

        except OSError:
            image = np.random.randint(0, 255, size=(50, 100), dtype=np.uint8)

        txt_name = self.txt_paths[idx]
        txt_filepath = os.path.join(self.txt_dir,txt_name)
        try:
            with open(txt_filepath,'r') as file:
                label = file.read()

        except OSError:

            label = ""
        if len(label) > self.max_len:
            self.max_len = len(label)

        sample = {'image': image, 'label': label}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample