# API Documentation

## 📚 Core Modules

### `dmfpga_redesign.py` - Complete DMFPGA Architecture

#### `DMFPGAModel`

Complete DMFPGA model implementing the full architecture pipeline following the research diagram.

```python
class DMFPGAModel(nn.Module):
    def __init__(self, fp_dim=1190, graph_dim=64, hidden_dim=128, num_heads=4, output_dim=1)
```

**Parameters:**
- `fp_dim` (int): Molecular fingerprint dimension (default: 1190)
- `graph_dim` (int): Graph feature dimension (default: 64)
- `hidden_dim` (int): Hidden layer dimension (default: 128)
- `num_heads` (int): Number of attention heads (default: 4)
- `output_dim` (int): Output dimension (default: 1)

**Architecture Components:**

1. **PLSRFeatureProcessor**: PLSR-inspired fingerprint processing
   ```python
   self.fp_processor = PLSRFeatureProcessor(fp_dim, hidden_dim)
   ```

2. **MolecularGraphAttention**: Multi-head graph attention network
   ```python
   self.graph_attention = MolecularGraphAttention(graph_dim, hidden_dim, num_heads)
   ```

3. **Feature Fusion**: Combines fingerprint and graph features
   ```python
   self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
   ```

4. **FCNN Classifier**: Progressive classification layers
   ```python
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
   ```

**Performance:**
- **AUC**: 97.67% ± 0.30%
- **Accuracy**: 89.58% ± 0.49%
- **Training time**: ~9-10 seconds per fold

#### `train_dmfpga_model(args, log)`

Complete training function for DMFPGA architecture.

**Parameters:**
- `args` (Namespace): DMFPGA training configuration
- `log` (Logger): Logger instance

**Returns:**
- `tuple`: (test_scores, val_scores, train_scores)

**Example:**
```python
from dmfpga_redesign import train_dmfpga_model, set_dmfpga_arguments, set_log

args = set_dmfpga_arguments()
log = set_log('dmfpga', 'dmfpga.log')
scores = train_dmfpga_model(args, log)
print(f"DMFPGA AUC: {scores[0][4]:.4f}")
```

#### `DMFPGADataset`

Enhanced dataset class for DMFPGA workflow.

```python
class DMFPGADataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train')
```

**Parameters:**
- `data_path` (str): Path to CSV file with molecular data
- `mode` (str): Dataset mode ('train', 'predict')

**Features:**
- Automatic SMILES extraction
- Molecular fingerprint processing (Extended + MACCS)
- Optimized for DMFPGA architecture

#### `set_dmfpga_arguments()`

Creates DMFPGA-specific training arguments.

**Returns:**
- `Namespace`: DMFPGA configuration object

**Default DMFPGA Values:**
```python
{
    'nhid': 128,          # Hidden dimension
    'graph_dim': 64,      # Graph feature dimension
    'nheads': 4,          # Graph attention heads
    'num_epochs': 30,     # Training epochs
    'batch_size': 32,     # Batch size
    'lr': 0.001,          # Learning rate
    'num_folds': 5,       # Cross-validation folds
    'seed': 42,           # Random seed
    'cuda': True          # GPU acceleration
}
```

### `dmfpga.data` - Baseline Components

#### `MoleDataSet`

Molecular dataset class for loading and preprocessing molecular data.

```python
class MoleDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, args=None, mode='train')
```

**Parameters:**
- `data_path` (str): Path to CSV file containing molecular data
- `args` (Namespace, optional): Training arguments
- `mode` (str): Dataset mode ('train', 'predict')

**Attributes:**
- `features` (np.ndarray): Molecular fingerprint features
- `labels` (np.ndarray): Target labels (for training mode)
- `smiles` (list): SMILES strings

**Methods:**

##### `__len__()`
Returns the number of samples in the dataset.

##### `__getitem__(idx)`
Returns a sample at the given index.

**Returns:**
- `tuple`: (smiles, features, labels) for training mode
- `tuple`: (smiles, features) for prediction mode

**Example:**
```python
from dmfpga.data import MoleDataSet

# Load training data
dataset = MoleDataSet("benchmark_embedded.csv", mode='train')
print(f"Dataset size: {len(dataset)}")
print(f"Feature shape: {dataset.features.shape}")

# Load prediction data
pred_dataset = MoleDataSet("new_molecules.csv", mode='predict')
```

---

### `dmfpga.train`

#### `fold_train(args, log)`

Trains a single fold of cross-validation.

**Parameters:**
- `args` (Namespace): Training configuration
  - `data_path` (str): Path to training data
  - `save_path` (str): Directory to save model
  - `nhid` (int): Hidden layer size
  - `dropout` (float): Dropout rate
  - `num_epochs` (int): Number of training epochs
  - `batch_size` (int): Batch size
  - `task_num` (int): Number of output tasks
  - `cuda` (bool): Use GPU if available
- `log` (Logger): Logger instance

**Returns:**
- `tuple`: (test_scores, val_scores, train_scores)
  - Each score array contains: [accuracy, precision, recall, specificity, auc]

**Example:**
```python
from dmfpga.train import fold_train
from dmfpga.tool import set_train_argument, set_log

args = set_train_argument()
args.data_path = "benchmark_embedded.csv"
args.save_path = "models/fold_1"
log = set_log('training', 'train.log')

test_scores, val_scores, train_scores = fold_train(args, log)
print(f"Test AUC: {test_scores[4]:.4f}")
```

#### `predict(model, dataset, batch_size, scaler)`

Makes predictions on new molecular data.

**Parameters:**
- `model` (torch.nn.Module): Trained PyTorch model
- `dataset` (MoleDataSet): Dataset for prediction
- `batch_size` (int): Batch size for inference
- `scaler` (StandardScaler): Fitted feature scaler

**Returns:**
- `list`: Prediction probabilities

**Example:**
```python
import torch
import joblib
from dmfpga.train import predict
from dmfpga.data import MoleDataSet

# Load model and scaler
model = torch.load("models/Seed_42/model.pt")
scaler = joblib.load("models/Seed_42/scaler.pkl")

# Load prediction data
dataset = MoleDataSet("new_molecules.csv", mode='predict')

# Make predictions
predictions = predict(model, dataset, batch_size=32, scaler=scaler)
```

---

### `dmfpga.tool`

#### `set_train_argument()`

Creates and returns training arguments with default values.

**Returns:**
- `Namespace`: Training configuration object

**Default Values:**
```python
{
    'seed': 42,
    'num_folds': 5,
    'metric': 'auc',
    'task_num': 1,
    'batch_size': 32,
    'cuda': False,
    'dataset_type': 'classification',
    'fp_2_dim': 500,
    'nhid': 40,
    'nheads': 7,
    'gat_scale': 0.4,
    'dropout': 0.35,
    'dropout_gat': 0.1,
    'num_epochs': 10
}
```

**Example:**
```python
from dmfpga.tool import set_train_argument

args = set_train_argument()
args.data_path = "my_data.csv"
args.save_path = "my_models/"
args.nhid = 64  # Override default
```

#### `set_log(name, log_path)`

Creates and configures a logger.

**Parameters:**
- `name` (str): Logger name
- `log_path` (str): Path to log file

**Returns:**
- `Logger`: Configured logger instance

**Example:**
```python
from dmfpga.tool import set_log

log = set_log('training', 'experiment.log')
log.info("Training started")
```

#### `mkdir(path)`

Creates directory if it doesn't exist.

**Parameters:**
- `path` (str): Directory path to create

**Example:**
```python
from dmfpga.tool import mkdir

mkdir("results/experiment_1")
```

---

## 🎯 Main Scripts

### `train.py`

Main training script with cross-validation.

**Command Line Arguments:**
```bash
python train.py \
  --data_path "data.csv" \
  --save_path "models/" \
  --log_path "train.log" \
  --nhid 40 \
  --dropout 0.35 \
  --num_epochs 10 \
  --batch_size 32 \
  --num_folds 5
```

### `predict.py`

Prediction script for new molecules.

**Command Line Arguments:**
```bash
python predict.py \
  --predict_path "molecules.smi" \
  --model_path "models/Seed_42" \
  --result_path "predictions.csv" \
  --batch_size 32 \
  --task_names "Hepato"
```

### `hyper_opti.py`

Hyperparameter optimization using Bayesian optimization.

**Command Line Arguments:**
```bash
python hyper_opti.py \
  --data_path "data.csv" \
  --save_path "hp_results/" \
  --log_path "hp.log" \
  --max_evals 50
```

---

## 🔧 Configuration Classes

### `TrainingArgs`

Training configuration namespace with the following attributes:

```python
class TrainingArgs:
    # Data
    data_path: str          # Path to training data
    save_path: str          # Model save directory
    log_path: str           # Log file path
    
    # Model Architecture
    nhid: int = 40          # Hidden layer size
    dropout: float = 0.35   # Dropout rate
    task_num: int = 1       # Number of output tasks
    
    # Training
    num_epochs: int = 10    # Training epochs
    batch_size: int = 32    # Batch size
    num_folds: int = 5      # CV folds
    seed: int = 42          # Random seed
    
    # Hardware
    cuda: bool = False      # Use GPU
    
    # Metrics
    metric: str = 'auc'     # Primary metric
    task_names: list = ['Hepato']  # Task names
```

---

## 📊 Data Formats

### Input CSV Format

```csv
smiles,label,ExtFP1,ExtFP2,...,ExtFP1024,MACCSFP1,...,MACCSFP166
CCO,1,0,1,0,...,1,1,...,0
CC(=O)O,0,1,0,1,...,0,0,...,1
```

**Required Columns:**
- `smiles`: SMILES string representation
- `label`: Binary target (0/1)
- `ExtFP1-ExtFP1024`: Extended fingerprint bits
- `MACCSFP1-MACCSFP166`: MACCS fingerprint bits

### SMILES Format (.smi)

```
CCO
CC(=O)O
c1ccccc1
```

One SMILES string per line.

### Prediction Output Format

```csv
smiles,prediction,probability
CCO,1,0.8234
CC(=O)O,0,0.1456
c1ccccc1,1,0.9123
```

---

## 🚀 Performance Utilities

### Model Loading

```python
import torch
import joblib

# Load complete model
model = torch.load("models/Seed_42/model.pt")

# Load model state dict (recommended)
model = create_model(input_size, hidden_size)
model.load_state_dict(torch.load("models/Seed_42/model_state.pt"))

# Load scaler
scaler = joblib.load("models/Seed_42/scaler.pkl")

# Load training args
args = joblib.load("models/Seed_42/train_args.pkl")
```

### Batch Prediction

```python
def batch_predict(model, smiles_list, scaler, batch_size=32):
    """Predict on a list of SMILES strings."""
    predictions = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        # Convert SMILES to features (implement your conversion)
        features = smiles_to_features(batch_smiles)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        with torch.no_grad():
            X = torch.tensor(features_scaled, dtype=torch.float32)
            pred = model(X).numpy()
            predictions.extend(pred.flatten())
    
    return predictions
```

### Model Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, dataset, scaler):
    """Comprehensive model evaluation."""
    predictions = predict(model, dataset, 32, scaler)
    
    # Convert probabilities to binary predictions
    pred_binary = [1 if p >= 0.5 else 0 for p in predictions]
    
    # Calculate metrics
    auc = roc_auc_score(dataset.labels, predictions)
    report = classification_report(dataset.labels, pred_binary)
    
    return {
        'auc': auc,
        'predictions': predictions,
        'binary_predictions': pred_binary,
        'report': report
    }
```

---

## 🔍 Error Handling

### Common Exceptions

```python
class DMFPGAError(Exception):
    """Base exception for DMFPGA."""
    pass

class DataLoadError(DMFPGAError):
    """Raised when data loading fails."""
    pass

class ModelError(DMFPGAError):
    """Raised when model operations fail."""
    pass

class PredictionError(DMFPGAError):
    """Raised when prediction fails."""
    pass
```

### Error Handling Examples

```python
try:
    dataset = MoleDataSet("data.csv", mode='train')
except FileNotFoundError:
    print("Data file not found!")
except Exception as e:
    print(f"Data loading error: {e}")

try:
    model = torch.load("model.pt")
except Exception as e:
    print(f"Model loading error: {e}")
```

---

## 🧪 Testing

### Unit Tests

```python
import unittest
from dmfpga.data import MoleDataSet

class TestMoleDataSet(unittest.TestCase):
    def test_dataset_loading(self):
        dataset = MoleDataSet("test_data.csv", mode='train')
        self.assertGreater(len(dataset), 0)
        
    def test_feature_shape(self):
        dataset = MoleDataSet("test_data.csv", mode='train')
        self.assertEqual(dataset.features.shape[1], 1190)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete training and prediction pipeline."""
    # Train model
    args = set_train_argument()
    args.data_path = "test_data.csv"
    args.save_path = "test_models/"
    args.num_epochs = 2  # Quick test
    
    log = set_log('test', 'test.log')
    scores = fold_train(args, log)
    
    # Load and test prediction
    model = torch.load("test_models/Seed_42/model.pt")
    scaler = joblib.load("test_models/Seed_42/scaler.pkl")
    
    pred_dataset = MoleDataSet("test_pred.csv", mode='predict')
    predictions = predict(model, pred_dataset, 32, scaler)
    
    assert len(predictions) == len(pred_dataset)
    assert all(0 <= p <= 1 for p in predictions)
```