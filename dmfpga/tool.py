import argparse
import os
import logging
import joblib
import torch

def mkdir(path):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)

def get_task_name(path):
    # """Lấy tên task từ đường dẫn."""
    # return os.path.basename(path)
    # lấy base name rồi bỏ phần extension, ví dụ:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name
def set_log(name, log_path):
    """Thiết lập logger ghi vào file log_path."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# def set_train_argument():
#     """Phân tích các tham số dòng lệnh cho training."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path',  type=str, required=True)
#     parser.add_argument('--save_path',  type=str, required=True)
#     parser.add_argument('--log_path',   type=str, required=True)
#     parser.add_argument('--seed',       type=int,   default=42)
#     parser.add_argument('--num_folds',  type=int,   default=5)
#     parser.add_argument('--metric',     type=str,   default='auc')
#     parser.add_argument('--task_num',   type=int,   default=1)
#     parser.add_argument('--task_names', nargs='+', default=['task'])
#     parser.add_argument('--batch_size', type=int,   default=32)
#     parser.add_argument('--cuda',       action='store_true')
#     parser.add_argument(
#         '--dataset_type',
#         type=str,
#         choices=['classification','regression'],
#         default='classification',
#         help='Type of task: classification (default) or regression'
#     )
#     # nếu cần thêm tham số, bổ sung tại đây
#     return parser.parse_args()
import argparse

def set_train_argument():
    """Phân tích các tham số dòng lệnh cho training."""
    parser = argparse.ArgumentParser()

    # Đường dẫn I/O
    parser.add_argument('--data_path',   type=str, required=True,
                        help='Path tới file CSV đầu vào')
    parser.add_argument('--save_path',   type=str, required=True,
                        help='Thư mục lưu kết quả và model')
    parser.add_argument('--log_path',    type=str, required=True,
                        help='File log cho quá trình training')

    # Thiết lập chung
    parser.add_argument('--seed',        type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_folds',   type=int, default=5,
                        help='Số fold cho cross‐validation')
    # parser.add_argument('--num_epochs',  type=int, default=10,
    #                     help='Số epoch cho mỗi fold')
    parser.add_argument('--metric',      type=str, default='auc',
                        help='Tên metric để đánh giá (ví dụ: auc, acc)')
    parser.add_argument('--task_num',    type=int, default=1,
                        help='Số output của model (1 nếu đơn nhiệm vụ)')
    parser.add_argument('--task_names',  nargs='+', default=['task'],
                        help='Tên cột label trong CSV')
    parser.add_argument('--batch_size',  type=int, default=32,
                        help='Kích thước batch')
    parser.add_argument('--cuda',        action='store_true',
                        help='Dùng GPU nếu có')

    # Loại bài toán
    parser.add_argument('--dataset_type',
                        choices=['classification','regression'],
                        default='classification',
                        help='Loại tác vụ: classification hoặc regression')

    # Hyper‐parameters (từ HyperOpt)
    parser.add_argument('--fp_2_dim',    type=int,   default=1024,
                        help='Độ dài fingerprint (nếu tính on‐the‐fly)')
    parser.add_argument('--nhid',        type=int,   default=64,
                        help='Số unit ẩn cho lớp fully‐connected')
    parser.add_argument('--nheads',      type=int,   default=1,
                        help='Số attention heads (nếu dùng GAT)')
    parser.add_argument('--gat_scale',   type=float, default=1.0,
                        help='Hệ số scale cho attention trong GAT')
    parser.add_argument('--dropout',     type=float, default=0.0,
                        help='Tỉ lệ dropout sau fully‐connected')
    parser.add_argument('--dropout_gat', type=float, default=0.0,
                        help='Tỉ lệ dropout bên trong GAT')
    parser.add_argument('--num_epochs',  type=int,   default=10,
                        help='Số epoch cho training')

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
