import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MoleDataSet(Dataset):
    """
    Dataset đơn giản:
      - nếu dùng fingerprint CSV: đọc file, giữ lại các cột fingerprint,
        cung cấp features và (khi training) label.
      - phương thức .smile() trả mảng các SMILES (để ghi file predict).
    """
    def __init__(self, csv_path, args=None, mode='predict'):
        df = pd.read_csv(csv_path)
        # Giả sử file predict cũng có cột 'Smiles'
        self.smiles = df['Smiles'].tolist()
        # Lấy toàn bộ cột khác ngoài 'Smiles'/'Label' làm features
        feats = df.drop(columns=['Smiles'] + (['Label'] if 'Label' in df else []))
        self.features = feats.values.astype(np.float32)
        # nếu training, có thêm label
        if mode == 'train' and 'Label' in df:
            # map chuỗi → số nếu cần
            self.label = df['Label'].tolist()
        else:
            self.label = None

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        X = self.features[idx]
        if self.label is not None:
            y = self.label[idx]
            return self.smiles[idx], X, y
        else:
            return self.smiles[idx], X

    def smile(self):
        """Trả về danh sách SMILES (dùng khi predict)."""
        return self.smiles
