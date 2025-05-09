import os
import torch
import joblib
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dmfpga.data import MoleDataSet

def fold_train(args, log):
    """
    Một fold training mẫu:
      1) Đọc CSV args.data_path (cột 'Smiles' + 'Label').
      2) Tạo MoleDataSet(mode='train'), DataLoader.
      3) Chuẩn hoá feature, lưu scaler và args.
      4) Khởi tạo model, optimizer, criterion.
      5) Chạy train loop đơn giản.
      6) Trả về các metric giả lập.
    """
    # 1) Load data
    df = pd.read_csv(args.data_path)
    dataset = MoleDataSet(args.data_path, mode='train')
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2) Fit scaler trên toàn bộ features
    all_X = dataset.features
    scaler = StandardScaler().fit(all_X)
    # 3) Lưu scaler và args
    os.makedirs(args.save_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.save_path, 'scaler.pkl'))
    joblib.dump(args,   os.path.join(args.save_path, 'train_args.pkl'))

    # 4) Khởi tạo model (người dùng thay bằng model thật)
    model = torch.nn.Sequential(
        torch.nn.Linear(all_X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, args.task_num),
        torch.nn.Sigmoid()
    )
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # 5) Train loop cơ bản
    for epoch in range(getattr(args, 'num_epochs', 10)):
        model.train()
        for item in loader:
            smiles, X, y = item
            X = torch.tensor(scaler.transform(X), device=device)
            y = torch.tensor(y, device=device).float()
            pred = model(X)
            loss = criterion(pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        gc.collect()

    # 6) Trả về dummy scores: [acc, precision, recall, specificity, auc]
    dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    return dummy, dummy, dummy

def predict(model, dataset, batch_size, scaler):
    """
    Dự đoán trên dataset (MoleDataSet, mode='predict'):
      - scaler: StandardScaler đã load
      - trả về list các giá trị (float) hoặc list[list] nếu đa nhiệm vụ.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []
    device = next(model.parameters()).device

    for item in loader:
        smi, X = item
        X = torch.tensor(scaler.transform(X), device=device)
        out = model(X)
        results.extend(out.detach().cpu().tolist())

    return results
