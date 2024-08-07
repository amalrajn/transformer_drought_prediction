import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def process_spi_sti_data(wdir):
    spi_files = [
        "/SPI/CONUS_SPI01.csv",
        "/SPI/CONUS_SPI03.csv",
        "/SPI/CONUS_SPI06.csv",
        "/SPI/CONUS_SPI09.csv",
        "/SPI/CONUS_SPI12.csv",
        "/SPI/CONUS_SPI18.csv",
        "/SPI/CONUS_SPI24.csv",
        "/SPI/CONUS_SPI36.csv",
        "/SPI/CONUS_SPI48.csv"
    ]
    
    sti_files = [
        "/STI/CONUS_STI01.csv",
        "/STI/CONUS_STI03.csv",
        "/STI/CONUS_STI06.csv",
        "/STI/CONUS_STI09.csv",
        "/STI/CONUS_STI12.csv",
        "/STI/CONUS_STI18.csv",
        "/STI/CONUS_STI24.csv"
    ]
    
    def load_and_process(file_list):
        combined = []
        for file in file_list:
            df = pd.read_csv(wdir + file, index_col=0)
            df = df.dropna(axis=1)  # Drop columns with NA values
            combined.append(df)
        return pd.concat(combined, axis=1)

    spi = load_and_process(spi_files)
    sti = load_and_process(sti_files)

    combined_data = pd.concat([spi, sti], axis=1)
    combined_data = combined_data.dropna()  # Drop rows with NaN values

    return combined_data

class SPIDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        return sample, sample  # Assuming input and target are the same for autoencoder

def create_dataloader(wdir, batch_size=32):
    processed_data = process_spi_sti_data(wdir)
    spi_dataset = SPIDataset(processed_data)
    dataloader = DataLoader(spi_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
