
import torch
from torch.utils.data import Dataset
from config import *
import json
import jieba
from PIL import Image
import torchvision.transforms as transforms
import os
from string import punctuation
import re


def encode_caption(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))


jieba.load_userdict('./data/user_dict.txt')

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, split, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.transform = transform
        self.punc = punctuation + u'um 1234567890'
        assert self.split in {'train', 'valid', 'test'}

        if split == 'train':
            with open(train_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:1]
        elif split == 'valid':
            with open(val_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:1]
        elif split == 'test':
            with open(test_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:1]
        with open(os.path.join(data_folder, 'WORDMAP_3_clear.json'), 'r', encoding='utf-8') as f:
            self.word_map = json.load(f)

    def __getitem__(self, index):
        # process image

        check_num = self.check_files[index].strip().split()[0]
        # image_files = os.listdir(os.path.join(image_folder, check_num))
        image_files = self.check_files[index].strip().split()[13:15]
        images = (self.transform(Image.open(os.path.join(image_folder, check_num, image_files[0]))),
                  self.transform(Image.open(os.path.join(image_folder, check_num, image_files[1]))),
                  )


        # process annotation
        diagnostic = self.check_files[index].strip().split()[1]
        diagnostic = re.sub(r"[{}]+".format(self.punc), "", diagnostic)
        diagnostic = list(jieba.cut(diagnostic))
        enc_diagnostic = encode_caption(self.word_map, diagnostic)
        caption = torch.LongTensor(enc_diagnostic)
        caplen = torch.LongTensor([len(diagnostic) + 2])

        return images, check_num, caption, caplen

    def __len__(self):
        return len(self.check_files)

if __name__ == '__main__':
    jieba.load_userdict('./data/user_dict.txt')













