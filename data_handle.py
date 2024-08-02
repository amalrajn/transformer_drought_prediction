import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx, :-1], dtype=torch.float32), torch.tensor(self.data[idx, -1], dtype=torch.float32)

def read_and_preprocess_data(wdir):
    # Read SPIs
    spi_files = ["CONUS_SPI01.csv", "CONUS_SPI03.csv", "CONUS_SPI06.csv", "CONUS_SPI09.csv", "CONUS_SPI12.csv",
                 "CONUS_SPI18.csv", "CONUS_SPI24.csv", "CONUS_SPI36.csv", "CONUS_SPI48.csv"]
    spi_data = [pd.read_csv(wdir + "/SPI/" + f, index_col=0).iloc[:, -120:] for f in spi_files]
    spi_data = [df.sort_index() for df in spi_data]

    # Read STIs
    sti_files = ["CONUS_STI01.csv", "CONUS_STI03.csv", "CONUS_STI06.csv", "CONUS_STI09.csv", "CONUS_STI12.csv",
                 "CONUS_STI18.csv", "CONUS_STI24.csv"]
    sti_data = [pd.read_csv(wdir + "/STI/" + f, index_col=0).iloc[:, -120:] for f in sti_files]
    sti_data = [df.sort_index() for df in sti_data]

    # Combine SPI and STI data
    combined_data = pd.concat(spi_data + sti_data, axis=1)
    return combined_data

def load_data(wdir, train_split=0.6, val_split=0.2, batch_size=32):
    data = read_and_preprocess_data(wdir)
    n_samples = len(data)
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    test_size = n_samples - train_size - val_size

    train_data = data[:train_size].values
    val_data = data[train_size:train_size + val_size].values
    test_data = data[train_size + val_size:train_size + val_size + test_size].values

    train_dataset = TimeSeriesDataset(train_data)
    val_dataset = TimeSeriesDataset(val_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader