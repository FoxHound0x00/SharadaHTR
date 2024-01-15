# Dataloader Class
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
# from SharadaDS import SharadaDataset

class SharadaDataLoader(object):

    def __init__(self, ds, batch_size=(16, 16), validation_split=0.2,
                 shuffle=True, seed=42, device='cpu'):
        assert isinstance(ds, SharadaDataset)
        assert isinstance(batch_size, tuple)
        assert isinstance(validation_split, float)
        assert isinstance(shuffle, bool)
        assert isinstance(seed, int)
        assert isinstance(device, str)

        self.ds = ds
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.device = device

    def  __call__(self):

        dataset_size = len(self.ds)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Dataloader
        train_loader = DataLoader(self.ds, batch_size=self.batch_size[0],
                                  sampler=train_sampler, collate_fn=self.collate_fn)
        validation_loader = DataLoader(self.ds, batch_size=self.batch_size[1],
                                       sampler=valid_sampler, collate_fn=self.collate_fn)

        return train_loader, validation_loader



    def collate_fn(self, batch):
        """Creates mini-batch tensors from the list of tuples (image, label).

        We should build custom collate_fn rather than using default collate_fn,
        because merging label tensor creates jagged array.
        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape (1, 128, 32).
                - label: torch tensor of shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, chan_in, height, width).
            targets: torch tensor of shape (sum(target_lengths)).
            lengths: torch tensor; length of each target label.
        """

        # Sort a data list by caption length (descending order).
        #sample.sort(key=lambda x: len(x[1]), reverse=True)
        images, labels = [b.get('image') for b in batch], [b.get('label') for b in batch]

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(label) for label in labels]
        targets = torch.zeros(sum(lengths)).long()
        lengths = torch.tensor(lengths)
        for j, label in enumerate(labels):
            start = sum(lengths[:j])
            end = lengths[j]
            targets[start:start+end] = torch.tensor([self.ds.char_dict.get(letter) for letter in label]).long()

        if self.device == 'cpu':
            dev = torch.device('cpu')
        else:
            dev = torch.device('cuda')

        return images.to(dev), targets.to(dev), lengths.to(dev)
