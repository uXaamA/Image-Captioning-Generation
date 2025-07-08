# Importing Required Libraries
import os
import re           # regular expression for searching and text patterns
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter   # to count how many times a class appears, useful for class balancing

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # itos -> index to string, <pad> for making sequence of same length  (Neural networks need equal-sized input to process them in batches)
        # <sos> start of sentence, <eos> end of sentence, <unk> for unseen words in the vocabulary
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v:k for k, v in self.itos.items()}   # stoi string to index
        # frequency threshold to set how many times a word must appear to be included in the vocab
        self.frequency_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)  # return how many tokens in the dictionary (vocabulary)
    

    def build_vocab (self, caption_list):
        frequencies = Counter()  # Counting the frequency of the words
        idx = 4        # Starting from the idx 4 as 0 to 3 we already define special tokens

        for caption in caption_list:
            for word in caption.split():
                frequencies[word] += 1

        # building filter based on frequency counter and building the vocab
        for word, freq in frequencies.items():
            if freq >= self.frequency_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1


class CaptionDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file, sep=',', header=0)      # reading the caption file
        self.transform = transform or self.default_transform()

        self.captions = self.df['caption'].apply(self.preprocess_text).tolist()    # preprocessing the text (caption), lowercase, removing special characters
        self.vocab = Vocabulary(freq_threshold)     # initializing vocabulary class
        self.vocab.build_vocab(self.captions)   # building the vocabulary using build_vocab function

    def default_transform(self):
            """ Resize the image to 224x224 and covert it to tensor
                Apply mean and std to normalize each channel"""
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
    def preprocess_text(self, text):
            """ Preprocess the text by lowercasing and removing special characters """
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)  # keeping only lowercas, spaces and removing special characters
            text = re.sub(r'\s+', ' ', text).strip()  # remove multiple spaces, strip() remove leading and trailing spaces
            return f'<sos> {text} <eos>'
        
    def __len__(self):
            """ Return the length of the caption file """
            return len(self.df)
        
    def __getitem__(self, idx):
            """ Get the image and caption for the given index and convert into numbers using vocab """
            # caption = self.captions[idx]   # pulls the preprocess caption for the image at index idx
            img_id = self.df.iloc[idx]['image']  # grab the image file name at index idx (cat12.jpg)
            img_captions = self.df[self.df['image'] == img_id]['caption'].tolist()
            caption = self.preprocess_text(np.random.choice(img_captions))
            image_path = os.path.join(self.root_dir, img_id)  # this will construct the full path for image
            image = Image.open(image_path).convert('RGB')  # Converting the image into RGB
            image = self.transform(image)   # applying the transformation

            # converting the caption into numbers using vocabulary class
            # for each word look up for the index in the vacabulary dictionary
            # if the word is not found in the vocab, it will return the index of <unk> token
            numerical_caption = [
                self.vocab.stoi.get(word, self.vocab.stoi['<unk'])
                for word in caption.split()  # turns the string into list of words
            ]

            return image, torch.tensor(numerical_caption)
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check if GPU is available
def collate_fn(batch):
    """ Collate function to pad the captions to the same length"""
    images, captions = zip(*batch) # unpacking the images and captions
    images = torch.stack(images) # stacking the images into a single tensor
    lengths = [len(cap) for cap in captions] # getting the length of each caption
    # taking all the captions and padding them to the same length match the length of the longest caption
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images.to(device), captions.to(device), torch.tensor(lengths)  




