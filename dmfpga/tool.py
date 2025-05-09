import argparse
import os
import logging
import joblib
import torch

def mkdir(path):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)

def get_task_name(path):
    """Lấy tên task từ đường dẫn."""
    return os.path.basename(path)

def set_log(name, log_path):
    """Thiết lập logger ghi vào file log_path."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def set_train_argument():
    """Phân tích các tham số dòng lệnh cho training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--save_path',  type=str, required=True)
    parser.add_argument('--log_path',   type=str, required=True)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--num_folds',  type=int,   default=5)
    parser.add_argument('--metric',     type=str,   default='auc')
    parser.add_argument('--task_num',   type=int,   default=1)
    parser.add_argument('--task_names', nargs='+', default=['task'])
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--cuda',       action='store_true')
    # nếu cần thêm tham số, bổ sung tại đây
    return parser.parse_args()

def set_predict_argument():
    """Phân tích các tham số dòng lệnh cho predicting."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, required=True)
    parser.add_argument('--model_path',   type=str, required=True)
    parser.add_argument('--result_path',  type=str, required=True)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--cuda',         action='store_true')
    parser.add_argument('--task_names',   nargs='+', required=True)
    return parser.parse_args()

def get_scaler(model_path):
    """Load scaler (StandardScaler...) đã lưu khi training."""
    return joblib.load(os.path.join(model_path, 'scaler.pkl'))

def load_args(model_path):
    """Load args (Namespace) đã lưu khi training."""
    return joblib.load(os.path.join(model_path, 'train_args.pkl'))

def load_model(model_path, use_cuda=False):
    """Load model đã lưu dạng Torch .pt."""
    fn = os.path.join(model_path, 'model.pt')
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = torch.load(fn, map_location=device)
    model.eval()
    return model

def load_data(predict_path, args):
    """Tạo dataset cho predicting."""
    from dmfpga.data import MoleDataSet
    return MoleDataSet(predict_path)
