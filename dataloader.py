from typing import AnyStr, List

import torch
from torch import Tensor


class DataLoader:
    def __init__(self, file_path: str, block_size: int, batch_size: int, train_split: int = 0.9):
        self.file_path = file_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_split = train_split

    def load_data_from_file(self) -> AnyStr:
        with open(self.file_path, 'r') as f:
            data: AnyStr = f.read()
        return data

    def create_token_encoder_and_decoder(self):
        chars_in_text: List = sorted(list(set(self.load_data_from_file())))
        vocab_size: int = len(chars_in_text)

        string_to_int: dict = {char: i for i, char in enumerate(chars_in_text)}
        int_to_string: dict = {i: char for i, char in enumerate(chars_in_text)}

        encoder = lambda s: [string_to_int[character] for character in s]
        decoder = lambda l: "".join([int_to_string[integer] for integer in l])

        return encoder, decoder, vocab_size

    def split_dataset(self):
        encoder, _, _ = self.create_token_encoder_and_decoder()
        data: Tensor = torch.tensor(encoder(self.load_data_from_file()))
        number_of_training_samples: int = int(self.train_split * len(data))

        train_data: Tensor = data[:number_of_training_samples]
        val_data: Tensor = data[number_of_training_samples:]

        return train_data, val_data

    def create_batches(self, split_type: str):
        train_data, val_data = self.split_dataset()
        if split_type == "train":
            data = train_data
        else:
            data = val_data

        idx: Tensor = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x: Tensor = torch.stack([data[i:i + self.block_size] for i in idx])
        y: Tensor = torch.stack([data[i + 1:i + self.block_size + 1] for i in idx])

        return x, y
