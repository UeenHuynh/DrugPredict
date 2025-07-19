# Usage Guide

## 🎯 Complete Workflow

### 1. Data Preparation

Your data should be in CSV format with molecular fingerprints:

```csv
smiles,label,ExtFP1,ExtFP2,...,MACCSFP1,MACCSFP2,...
CCO,1,0,1,0,...,1,0,1,...
CC(=O)O,0,1,0,1,...,0,1,0,...
```

### 2. DMFPGA Complete Training (Recommended)

```bash
# Run complete DMFPGA architecture
python dmfpga_redesign.py
```

### 3. Basic Baseline Training

```bash
python train.py \
  --data_path "your_data.csv" \
  --save_path models/ \
  --log_path training.log \
  --nhid 40 \
  --dropout 0.35 \
  --num_epochs 10
```

### 3. Hyperparameter Optimization

```bash
python hyper_opti.py \
  --data_path "your_data.csv" \
  --save_path hp_results/ \
  --log_path hp.log \
  --max_evals 100
```

### 4. Model Evaluation

Check the DMFPGA log file for results:
```bash
tail -20 dmfpga.log
```

Expected output (DMFPGA Complete):
```
INFO: 🎯 FINAL DMFPGA RESULTS
INFO: Accuracy:    0.8958 ± 0.0049
INFO: Precision:   0.8892 ± 0.0302
INFO: Recall:      0.9276 ± 0.0372
INFO: Specificity: 0.8573 ± 0.0472
INFO: AUC:         0.9767 ± 0.0030
```

For baseline model:
```bash
tail -20 training.log
```

Expected output (Baseline):
```
INFO: test auc = 0.876626
INFO: Average train auc = 0.829474  acc = 0.861970
INFO: Average val auc = 0.942243  acc = 0.861970
```

### 5. Making Predictions

```bash
python predict.py \
  --predict_path new_molecules.smi \
  --model_path models/Seed_42 \
  --result_path predictions.csv
```

## 🔧 Parameter Tuning Guide

### DMFPGA Complete Parameters

| Parameter | Description | Optimal Value | Range |
|-----------|-------------|---------------|-------|
| `nhid` | Hidden dimension | 128 | 64-256 |
| `graph_dim` | Graph feature dim | 64 | 32-128 |
| `nheads` | Attention heads | 4 | 2-8 |
| `num_epochs` | Training epochs | 30 | 20-50 |
| `batch_size` | Batch size | 32 | 16-64 |
| `lr` | Learning rate | 0.001 | 0.0001-0.01 |

### Baseline Parameters

| Parameter | Description | Optimal Value | Range |
|-----------|-------------|---------------|-------|
| `nhid` | Hidden layer size | 40 | 20-128 |
| `dropout` | Dropout rate | 0.35 | 0.1-0.5 |
| `num_epochs` | Training epochs | 10 | 5-50 |
| `batch_size` | Batch size | 32 | 16-128 |

### Performance Comparison

- **DMFPGA Complete**: 97.7% AUC 🔥 **BEST**
- **Simple Baseline**: 87.7% AUC ⭐ **Fast & Good**
- **Complex without attention**: 65-70% AUC (overfitting)

## 📊 Interpreting Results

### Metrics Explanation

- **AUC**: Area Under ROC Curve (0.5-1.0, higher is better)
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Specificity**: True Negatives / (True Negatives + False Positives)

### Good Performance Indicators

✅ **Excellent**: AUC > 0.85, Accuracy > 0.80
✅ **Good**: AUC > 0.75, Accuracy > 0.70
⚠️ **Fair**: AUC > 0.65, Accuracy > 0.60
❌ **Poor**: AUC < 0.65, Accuracy < 0.60

## 🚨 Common Pitfalls

### 1. Overfitting
**Symptoms**: High training accuracy, low validation accuracy
**Solutions**:
- Increase dropout (0.3 → 0.4)
- Reduce model size (nhid 64 → 40)
- Add early stopping

### 2. Underfitting
**Symptoms**: Low training and validation accuracy
**Solutions**:
- Increase model size (nhid 40 → 64)
- Reduce dropout (0.4 → 0.3)
- Increase epochs (10 → 20)

### 3. Data Issues
**Symptoms**: Inconsistent results across folds
**Solutions**:
- Check data quality
- Remove outliers
- Balance dataset

## 🎛️ Advanced Configuration

### Custom Model Architecture

Edit `dmfpga/train.py`:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(all_X.shape[1], args.nhid),
    torch.nn.BatchNorm1d(args.nhid),  # Add batch norm
    torch.nn.ReLU(),
    torch.nn.Dropout(args.dropout),
    torch.nn.Linear(args.nhid, args.nhid // 2),  # Add layer
    torch.nn.ReLU(),
    torch.nn.Linear(args.nhid // 2, args.task_num),
    torch.nn.Sigmoid()
)
```

### Custom Loss Functions

```python
# Focal Loss for imbalanced data
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Use in training
criterion = FocalLoss(alpha=1, gamma=2)
```

## 📈 Performance Optimization

### Speed Optimization

1. **Use GPU**: Add `--cuda` flag
2. **Increase batch size**: `--batch_size 64`
3. **Reduce precision**: Use `torch.float16`

### Memory Optimization

1. **Gradient accumulation**:
```python
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

2. **Clear cache**:
```python
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

## 🔍 Debugging Guide

### Check Data Loading

```python
from dmfpga.data import MoleDataSet
dataset = MoleDataSet("your_data.csv", mode='train')
print(f"Dataset size: {len(dataset)}")
print(f"Feature shape: {dataset.features.shape}")
print(f"Label distribution: {np.bincount(dataset.labels)}")
```

### Monitor Training

```python
# Add to training loop
print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
if epoch % 5 == 0:
    # Evaluate on validation set
    val_acc = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.4f}")
```

### Visualize Results

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

## 🎯 Best Practices

1. **Always use cross-validation** for reliable results
2. **Start with simple models** before adding complexity
3. **Monitor both training and validation metrics**
4. **Save models regularly** during training
5. **Document hyperparameters** for reproducibility
6. **Use version control** for code and data
7. **Test on held-out data** for final evaluation

## 📞 Getting Help

If you encounter issues:

1. Check the log files for error messages
2. Verify data format and preprocessing
3. Try with default hyperparameters first
4. Reduce model complexity if overfitting
5. Open an issue with detailed error information