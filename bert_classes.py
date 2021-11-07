# DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import *

# utilities
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm_notebook
import seaborn as sns

# helper functions
import utils

class IMDBDataset(Dataset):
    """

    """
    def __init__(self, reviews, sentiments, tokenizer, max_len):
        """

        :param reviews:
        :param sentiments:
        :param tokenizer:
        :param max_len:
        """
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """

        :return:
        """
        return len(self.reviews)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        review = str(self.reviews[item])
        sentiment = self.sentiments[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=200,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiments': torch.tensor(sentiment, dtype=torch.long)
        }