import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import *


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
            max_length=16,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiments': torch.tensor(sentiment, dtype=torch.long)
        }

class IMDBClassifier(nn.Module):
    """

    """

    def __init__(self, n_classes, model):
        """

        :param n_classes:
        :param model:
        """
        super(IMDBClassifier, self).__init__()
        self.bert = model['model'].from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """

        :param input_ids:
        :param attention_mask:
        :return:
        """
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)
