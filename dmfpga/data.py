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
    # def __init__(self, csv_path, args=None, mode='predict'):
    #     df = pd.read_csv(csv_path)
    #     # Giả sử file predict cũng có cột 'Smiles'
    #     self.smiles = df['Smiles'].tolist()
    #     # nếu header là 'smiles' hoặc 'SMILES', lấy linh hoạt:
    #     if 'Smiles' in df.columns:
    #         key = 'Smiles'
    #     elif 'smiles' in df.columns:
    #         key = 'smiles'
    #     else:
    #         raise KeyError("Không tìm thấy cột Smiles hoặc smiles trong data")
    #     self.smiles = df[key].tolist()
    #     # Lấy toàn bộ cột khác ngoài 'Smiles'/'Label' làm features
    #     feats = df.drop(columns=['Smiles'] + (['Label'] if 'Label' in df else []))
    #     self.features = feats.values.astype(np.float32)
    #     # nếu training, có thêm label
    #     if mode == 'train' and 'Label' in df:
    #         # map chuỗi → số nếu cần
    #         self.label = df['Label'].tolist()
    #     else:
    #         self.label = None
    def __init__(self, csv_path, args=None, mode='predict'):
        import pandas as pd
        import numpy as np

        df = pd.read_csv(csv_path)

        # 1) Xác định cột SMILES
        if 'Smiles' in df.columns:
            key_smiles = 'Smiles'
        elif 'smiles' in df.columns:
            key_smiles = 'smiles'
        else:
            raise KeyError("Không tìm thấy cột 'Smiles' hoặc 'smiles' trong CSV")
        self.smiles = df[key_smiles].tolist()

        # 2) Xác định cột nhãn (chỉ khi mode='train')
        label_col = None
        if mode == 'train':
            # Ví dụ mặc định bạn dùng 'Label'; nếu khác thì thêm điều kiện
            if 'label' in df.columns:
                label_col = 'label'
            elif 'activity' in df.columns:
                label_col = 'activity'
            else:
                raise KeyError("Không tìm thấy cột nhãn cho training")
            self.label = df[label_col].tolist()
        else:
            self.label = None

        # 3) Tạo features bằng cách drop cột smiles và label (nếu có)
        drop_cols = [key_smiles]
        if label_col:
            drop_cols.append(label_col)
        feats = df.drop(columns=drop_cols)
        self.features = feats.values.astype(np.float32)

    def __len__(self):
        # Độ dài dataset = số dòng
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        x = self.features[idx]       # numpy array, có thể float64
        y = self.label[idx]          # int hoặc float
        # Ép cả x và y về đúng dtype
        x_t = torch.from_numpy(x).float()
        y_t = torch.tensor(y, dtype=torch.long)  # hoặc float nếu regression

        # 3) Lấy label (None nếu mode != 'train')
        if self.label is not None:
            # classification: long; regression: float
            dtype = torch.long if isinstance(self.label[0], int) else torch.float
            y_t = torch.tensor(self.label[idx], dtype=dtype)
        else:
            y_t = None

        return smi, x_t, y_t
    # def __getitem__(self, idx):
    #     X = self.features[idx]
    #     if self.label is not None:
    #         y = self.label[idx]
    #         return self.smiles[idx], X, y
    #     else:
    #         return self.smiles[idx], X

