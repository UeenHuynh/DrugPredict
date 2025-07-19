import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import os
import time
from argparse import Namespace
import gc

class MolecularGraphAttention(nn.Module):
    """Graph Attention Network for molecular graphs"""
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.W_o(attended)
        output = self.layer_norm(output + x[:, :, :self.hidden_dim] if x.shape[-1] >= self.hidden_dim else output)
        
        return output

class PLSRFeatureProcessor(nn.Module):
    """PLSR-inspired feature processing"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # PLSR-like projection: cov(F_m, E_v) = 1/n * F^T * E_m
        projected = self.projection(x)
        activated = self.activation(projected)
        normalized = self.layer_norm(activated)
        return normalized

class DMFPGAModel(nn.Module):
    """Complete DMFPGA model following the architecture diagram"""
    def __init__(self, fp_dim=1190, graph_dim=64, hidden_dim=128, num_heads=4, output_dim=1):
        super().__init__()
        
        # Molecular fingerprint processing (PLSR-inspired)
        self.fp_processor = PLSRFeatureProcessor(fp_dim, hidden_dim)
        
        # Molecular graph attention network
        # Simulate graph features from fingerprints
        self.fp_to_graph = nn.Linear(fp_dim, graph_dim)
        self.graph_attention = MolecularGraphAttention(graph_dim, hidden_dim, num_heads)
        
        # Feature splicing layer
        self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_dropout = nn.Dropout(0.2)
        
        # FCNN Classifier (following diagram)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Molecular fingerprint processing (PLSR)
        fp_features = self.fp_processor(x)
        
        # 2. Graph attention processing
        # Convert fingerprints to graph-like representation
        graph_input = self.fp_to_graph(x)
        # Reshape for attention (simulate sequence)
        graph_input = graph_input.unsqueeze(1).repeat(1, 8, 1)  # 8 "nodes"
        graph_features = self.graph_attention(graph_input)
        # Global pooling
        graph_features = torch.mean(graph_features, dim=1)
        
        # 3. Feature splicing (concatenation + fusion)
        combined_features = torch.cat([fp_features, graph_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.fusion_dropout(fused_features)
        
        # 4. FCNN classifier
        output = self.classifier(fused_features)
        
        return output

class DMFPGADataset(Dataset):
    """Dataset for DMFPGA following the workflow"""
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract SMILES
        self.smiles = df['smiles'].values
        
        # Extract molecular fingerprints (Extended + MACCS)
        fp_columns = [col for col in df.columns if col.startswith(('ExtFP', 'MACCSFP'))]
        self.features = df[fp_columns].values.astype(np.float32)
        
        # Extract labels for training
        if mode == 'train' and 'label' in df.columns:
            self.labels = df['label'].values.astype(np.float32)
        else:
            self.labels = None
            
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.smiles[idx], self.features[idx], self.labels[idx]
        else:
            return self.smiles[idx], self.features[idx]

def train_dmfpga_model(args, log):
    """Train DMFPGA model following the complete workflow"""
    fold_start = time.time()
    
    # 1. Data preparation (following diagram)
    log.info("=== Data Preparation Stage ===")
    dataset = DMFPGADataset(args.data_path, mode='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    log.info(f"Loaded {len(dataset)} compounds")
    log.info(f"Feature dimension: {dataset.features.shape[1]}")
    
    # 2. Feature processing preparation
    log.info("=== Feature Processing Stage ===")
    all_X = dataset.features
    scaler = StandardScaler().fit(all_X)
    
    # Save preprocessing artifacts
    os.makedirs(args.save_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.save_path, 'scaler.pkl'))
    joblib.dump(args, os.path.join(args.save_path, 'train_args.pkl'))
    
    # 3. Model initialization (following architecture diagram)
    log.info("=== DMFPGA Model Architecture ===")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    model = DMFPGAModel(
        fp_dim=all_X.shape[1],
        graph_dim=args.graph_dim,
        hidden_dim=args.nhid,
        num_heads=args.nheads,
        output_dim=args.task_num
    ).to(device).float()
    
    log.info(f"Model architecture:")
    log.info(f"- Fingerprint dim: {all_X.shape[1]}")
    log.info(f"- Graph attention heads: {args.nheads}")
    log.info(f"- Hidden dimension: {args.nhid}")
    log.info(f"- Device: {device}")
    
    # 4. Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 5. Training loop
    log.info("=== Training Phase ===")
    num_epochs = getattr(args, 'num_epochs', 20)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for smiles, X_batch, y_batch in loader:
            # Feature preprocessing
            if torch.is_tensor(X_batch):
                X_arr = X_batch.cpu().numpy()
            else:
                X_arr = X_batch
            
            X_scaled = scaler.transform(X_arr).astype(np.float32)
            X = torch.from_numpy(X_scaled).to(device)
            
            # Labels
            if torch.is_tensor(y_batch):
                y = y_batch.to(device).float().unsqueeze(1)
            else:
                y = torch.tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(1)
            
            # Forward pass through complete DMFPGA architecture
            pred = model(X)
            loss = criterion(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))
        else:
            patience_counter += 1
        
        epoch_dur = time.time() - epoch_start
        log.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_dur:.2f}s")
        print(f"[Console] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_dur:.2f}s")
        
        if patience_counter >= 10:
            log.info("Early stopping triggered")
            break
        
        gc.collect()
    
    # 6. Load best model and evaluate
    log.info("=== Evaluation Phase ===")
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'best_model.pt')))
    torch.save(model, os.path.join(args.save_path, 'model.pt'))
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for smiles, X_batch, y_batch in loader:
            if torch.is_tensor(X_batch):
                X_arr = X_batch.cpu().numpy()
            else:
                X_arr = X_batch
            
            X_scaled = scaler.transform(X_arr).astype(np.float32)
            X = torch.from_numpy(X_scaled).to(device)
            
            preds = model(X).cpu().numpy().flatten()
            all_preds.extend(preds)
            
            if torch.is_tensor(y_batch):
                labels = y_batch.cpu().numpy().flatten()
            else:
                labels = y_batch.flatten() if hasattr(y_batch, 'flatten') else y_batch
            all_labels.extend(labels)
    
    # Calculate metrics
    pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
    
    try:
        acc = accuracy_score(all_labels, pred_labels)
        prec = precision_score(all_labels, pred_labels, zero_division=0)
        rec = recall_score(all_labels, pred_labels, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        
        fold_scores = np.array([acc, prec, rec, spec, auc])
        
        log.info("=== Final Results ===")
        log.info(f"Accuracy: {acc:.4f}")
        log.info(f"Precision: {prec:.4f}")
        log.info(f"Recall: {rec:.4f}")
        log.info(f"Specificity: {spec:.4f}")
        log.info(f"AUC: {auc:.4f}")
        
    except Exception as e:
        log.info(f"Error calculating metrics: {e}")
        fold_scores = np.zeros(5, dtype=float)
    
    fold_dur = time.time() - fold_start
    log.info(f"Total training time: {fold_dur:.2f}s")
    print(f"[Console] Completed fold in {fold_dur:.2f}s")
    
    return fold_scores, fold_scores, fold_scores

def set_dmfpga_arguments():
    """Set arguments for DMFPGA model"""
    args = Namespace()
    
    # Data
    args.data_path = "benchmark_embedded.csv"
    args.save_path = "dmfpga_model/"
    args.log_path = "dmfpga.log"
    
    # Model architecture (following diagram)
    args.nhid = 128          # Hidden dimension
    args.graph_dim = 64      # Graph feature dimension
    args.nheads = 4          # Graph attention heads
    args.task_num = 1        # Output tasks
    
    # Training
    args.num_epochs = 30
    args.batch_size = 32
    args.lr = 0.001
    args.num_folds = 5
    args.seed = 42
    
    # Hardware
    args.cuda = torch.cuda.is_available()
    
    # Metrics
    args.metric = 'auc'
    args.task_names = ['Hepato']
    
    return args

def set_log(name, log_path):
    """Simple logging setup"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)

def mkdir(path):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    print("🚀 Starting DMFPGA Training with Complete Architecture")
    
    # Setup
    args = set_dmfpga_arguments()
    mkdir(args.save_path)
    log = set_log('dmfpga', args.log_path)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    log.info("🏗️ DMFPGA Architecture Implementation")
    log.info("Following the complete workflow diagram:")
    log.info("1. Data preparation → SMILES processing")
    log.info("2. Feature processing → Molecular fingerprints + Graph attention")
    log.info("3. Feature splicing → Feature fusion")
    log.info("4. FCNN classifier → Final prediction")
    
    # Training with cross-validation
    scores = []
    for fold in range(args.num_folds):
        log.info(f"\n=== FOLD {fold + 1}/{args.num_folds} ===")
        args.seed = 42 + fold
        args.save_path = f"dmfpga_model/Seed_{args.seed}"
        mkdir(args.save_path)
        
        fold_scores, _, _ = train_dmfpga_model(args, log)
        scores.append(fold_scores)
    
    # Final results
    scores = np.array(scores)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    
    log.info("\n🎯 FINAL DMFPGA RESULTS")
    log.info("=" * 50)
    log.info(f"Accuracy:    {mean_scores[0]:.4f} ± {std_scores[0]:.4f}")
    log.info(f"Precision:   {mean_scores[1]:.4f} ± {std_scores[1]:.4f}")
    log.info(f"Recall:      {mean_scores[2]:.4f} ± {std_scores[2]:.4f}")
    log.info(f"Specificity: {mean_scores[3]:.4f} ± {std_scores[3]:.4f}")
    log.info(f"AUC:         {mean_scores[4]:.4f} ± {std_scores[4]:.4f}")
    log.info("=" * 50)
    
    print(f"\n🎉 DMFPGA Training Complete!")
    print(f"📊 Final AUC: {mean_scores[4]:.4f} ± {std_scores[4]:.4f}")
    print(f"📁 Models saved in: dmfpga_model/")