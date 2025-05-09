import os
import time
import torch
import joblib
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dmfpga.data import MoleDataSet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
def fold_train(args, log):
    fold_start = time.time()

    # 1) Load data
    dataset = MoleDataSet(args.data_path, args=args, mode='train')
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2) Fit scaler & save
    all_X = dataset.features
    scaler = StandardScaler().fit(all_X)
    os.makedirs(args.save_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.save_path, 'scaler.pkl'))
    joblib.dump(args,   os.path.join(args.save_path, 'train_args.pkl'))

    # 3) Build model
    model = torch.nn.Sequential(
        torch.nn.Linear(all_X.shape[1], args.nhid),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.nhid, args.task_num),
        torch.nn.Sigmoid()
    ).to(device := (torch.device('cuda') if args.cuda and torch.cuda.is_available()
                    else torch.device('cpu'))).float()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # 5) Train loop cơ bản (đã sửa)
    num_epochs = getattr(args, 'num_epochs', 10)
    for epoch in range(num_epochs):
        # 5.1) Start epoch
        epoch_start = time.time()
        log.info(f"[{args.save_path}] Starting epoch {epoch+1}/{num_epochs}")
        print(f"[Console] Starting epoch {epoch+1}/{num_epochs}")

        model.train()
        for smiles, X_batch, y_batch in loader:
            # Chuẩn hoá & cast X
            if torch.is_tensor(X_batch):
                X_arr = X_batch.cpu().numpy()
            else:
                X_arr = X_batch
            X_scaled = scaler.transform(X_arr).astype(np.float32)
            X = torch.from_numpy(X_scaled).to(device)

            # Cast y
            if torch.is_tensor(y_batch):
                y = y_batch.to(device).float().unsqueeze(1)
            else:
                y = torch.tensor(
                    y_batch, dtype=torch.float32, device=device
                ).unsqueeze(1)

            # Forward / backward
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gc.collect()

        # 5.2) Log thời gian mỗi epoch
        epoch_dur = time.time() - epoch_start
        log.info(f"[{args.save_path}] Epoch {epoch+1} done in {epoch_dur:.2f}s")
        print(f"[Console] Epoch {epoch+1} done in {epoch_dur:.2f}s")

    # 6) Log tổng thời gian fold
    fold_dur = time.time() - fold_start
    log.info(f"[{args.save_path}] Completed fold in {fold_dur:.2f}s")
    print(f"[Console] Completed fold in {fold_dur:.2f}s")

    # 7) Trả về dummy (thay real metric sau này)
    dummy = np.zeros(5, dtype=float)
    return dummy, dummy, dummy
    # """
    # Một fold training mẫu:
    #   1) Đọc CSV args.data_path (cột 'Smiles' + 'Label').
    #   2) Tạo MoleDataSet(mode='train'), DataLoader.
    #   3) Chuẩn hoá feature, lưu scaler và args.
    #   4) Khởi tạo model, optimizer, criterion.
    #   5) Chạy train loop đơn giản.
    #   6) Trả về các metric giả lập.
    # """
    # fold_start = time.time()
    # # 1) Load data
    # df = pd.read_csv(args.data_path)
    # dataset = MoleDataSet(args.data_path, mode='train')
    # loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # # 2) Fit scaler trên toàn bộ features
    # all_X = dataset.features
    # scaler = StandardScaler().fit(all_X)
    # # 3) Lưu scaler và args
    # os.makedirs(args.save_path, exist_ok=True)
    # joblib.dump(scaler, os.path.join(args.save_path, 'scaler.pkl'))
    # joblib.dump(args,   os.path.join(args.save_path, 'train_args.pkl'))

    # # 4) Khởi tạo model (người dùng thay bằng model thật)
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(all_X.shape[1], 64),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(64, args.task_num),
    #     torch.nn.Sigmoid()
    # )
    # device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # optimizer = torch.optim.Adam(model.parameters())
    # criterion = torch.nn.BCELoss()

    # # 5) Train loop cơ bản
    # num_epochs = getattr(args, 'num_epochs', 10)
    # for epoch in range(getattr(args, 'num_epochs', 10)):
    #     epoch_start = time.time()
    #     log.info(f"[{args.save_path}] Starting epoch {epoch+1}/{num_epochs}")
    #     model.train()
    #     # for item in loader:
    #     #     smiles, X, y = item
    #     #     X = torch.tensor(scaler.transform(X), device=device)
    #     #     y = torch.tensor(y, device=device).float()
    #     #     pred = model(X)
    #     #     loss = criterion(pred, y.unsqueeze(1))
    #     #     optimizer.zero_grad()
    #     #     loss.backward()
    #     #     optimizer.step()
    #     # gc.collect()
    #     # for smiles, X_np, y_np in loader:
    #     #     # — scale and cast features to float32
    #     #     X_scaled = scaler.transform(X_np)              
    #     #     X = torch.from_numpy(X_scaled.astype(np.float32)).to(device)

    #     #     # — cast labels to float32 (for BCELoss)
    #     #     y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
    #     for smiles, X_batch, y_batch in loader:
    #         # --- 5.1) Chuẩn hóa và cast dtype cho X ---
    #         # nếu X_batch là Tensor, chuyển về numpy
    #         if torch.is_tensor(X_batch):
    #             X_arr = X_batch.cpu().numpy()
    #         else:
    #             X_arr = X_batch
    #         X_scaled = scaler.transform(X_arr)            # numpy float64
    #         # cast về float32 rồi đưa về device
    #         X = torch.from_numpy(X_scaled.astype(np.float32)).to(device)

    #         # --- 5.2) Cast y về float32 luôn trên device ---
    #         if torch.is_tensor(y_batch):
    #             y = y_batch.to(device).float().unsqueeze(1)
    #         else:
    #             y = torch.tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(1)

    #         pred = model(X)
    #         loss = criterion(pred, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     gc.collect()
    #     dur = time.time() - epoch_start
    #     log.info(f"[{args.save_path}] Epoch {epoch+1} done in {dur:.2f}s")

    # # 5) Fold done
    # total_dur = time.time() - fold_start
    # log.info(f"[{args.save_path}] Completed fold in {total_dur:.2f}s")
    # # 6) Trả về dummy scores: [acc, precision, recall, specificity, auc]
    # dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # return dummy, dummy, dummy

    # model.eval()
    # all_preds = []
    # all_labels = []
    # val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # with torch.no_grad():
    #     for smiles, X_np, y_np in val_loader:
    #         X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    #         preds = model(X).cpu().numpy().flatten()
    #         all_preds.extend(preds)
    #         all_labels.extend(y_np)

    # # 7) Tính metric
    # # classification threshold = 0.5
    # pred_labels = [1 if p>=0.5 else 0 for p in all_preds]
    # acc   = accuracy_score(all_labels, pred_labels)
    # prec  = precision_score(all_labels, pred_labels, zero_division=0)
    # rec   = recall_score(all_labels, pred_labels, zero_division=0)
    # tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
    # spec  = tn / (tn+fp) if (tn+fp)>0 else 0.0
    # auc   = roc_auc_score(all_labels, all_preds)

    # # 8) Trả về array [acc, prec, rec, spec, auc] và độ lệch chuẩn nếu tính nhiều fold
    # fold_scores = np.array([acc, prec, rec, spec, auc])
    # # ở đây chỉ có 1 fold so dummy bạn cần return fold_scores và np.zeros(5) cho std
    # return fold_scores, np.zeros(5), np.zeros(5)
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
        # X = torch.tensor(scaler.transform(X), device=device)
        X = torch.tensor(scaler.transform(X),
                dtype=torch.float32,
                device=device)
        out = model(X)
        results.extend(out.detach().cpu().tolist())

    return results
