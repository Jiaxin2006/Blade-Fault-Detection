# save as: plot_embeddings_ot_vs_temp.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
import torch

sns.set(style="whitegrid")

# 数据/模型超参
SEQ_LEN = 4                 # 输入序列长度
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
DROPOUT_RATE = 0.0

CNN_CHANNELS = 16
CNN_KERNEL = 3
LSTM_HID = 128
TRANS_DMODEL = 128
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 0

VAL_RATIO = 0.10             # VERY IMPORTANT: 验证集不为0
TEST_RATIO = 0.20

# 训练策略/损失 可选开关
USE_HETEROSCEDASTIC = True  # True: 输出 (mu, logvar) + Gaussian NLL
USE_ASYMMETRIC_LOSS = True    # True: 低估加重惩罚
ASYM_UNDER_WEIGHT = 2.0       # 低估惩罚倍数（diff<0）
OVERSAMPLE_PEAKS = False      # True: 训练集对峰值过采样
PEAK_PERCENTILE = 95          # 以训练集label的95分位作为峰值阈值
PEAK_WEIGHT_ALPHA = 2.0       # 过采样时峰值样本权重

PATIENCE = 3                  # 学习率调度等待
LR_FACTOR = 0.5

# ----------------- CONFIG -----------------
OUT_DIR = Path("out_cnn_lstm_cluster_1/")  # 与训练输出目录一致
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "标注的数据-#67_1.xlsx"   # 原始数据（same as in your main script）
SEQ_LEN = 4                         # must match training SEQ_LEN
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your global (without-cluster) saved model state_dict
# 如果你没有 global model，请把下面改为 cluster 模型路径，例如:
# GLOBAL_MODEL_PATH = OUT_DIR/"model_cnn_lstm_att_final_0.pt"
GLOBAL_MODEL_PATH = OUT_DIR / "model_run0_cluster0.pt"
# fallback if not exist:
FALLBACK_MODEL = OUT_DIR / "model_cnn_lstm_att_final_0.pt"

# Name of saved scaler (fit on train) — 用于和原脚本保持一致
SCALER_PATH = OUT_DIR / "scaler_inputs.joblib"

# output files
EMB_NPY = OUT_DIR / "test_embeddings.npy"
META_CSV = OUT_DIR / "test_embeddings_meta.csv"
SCATTER_PNG = OUT_DIR / "scatter_ot_temp_by_emb_cluster-2.png"

# ----------------- Load data -----------------
# ------------------ READ & PREPROCESS ------------------
print("Reading data...")
data_path = "标注的数据-#67_1.xlsx"
df_raw = pd.read_excel(data_path)

# 更鲁棒的候选列名（包含中英文多种写法）
col_candidates = {
    'time': ['time','timestamp','date','统计时间','时间','datetime'],
    'OT': ['OT','ot','output','efficiency','目标','目标值','功率','power'],
    'exog_temp': ['exog_temp','Exogenous1','temperature','temp','外温','温度','气温','环境温度'],
    'exog_wind': ['exog_wind','Exogenous2','wind_speed','wind','风速','风速m/s']
}

# 尝试匹配列名（大小写不敏感）
cols = {}
columns_lower = {c.lower(): c for c in df_raw.columns}  # map lower->original
for logical, candidates in col_candidates.items():
    for cand in candidates:
        lc = cand.lower()
        # 先精确匹配候选（忽略大小写）
        if lc in columns_lower:
            cols[logical] = columns_lower[lc]
            break
    if logical not in cols:
        # 再做模糊通过子串匹配（比如 '温度' 在 '环境温度' 中）
        for orig in df_raw.columns:
            lorig = orig.lower()
            for cand in candidates:
                if cand.lower() in lorig:
                    cols[logical] = orig
                    break
            if logical in cols:
                break

# 如果仍然缺失必需列，打印列名提示用户并抛出异常
required = ['time','OT','exog_temp']  # exog_wind 非必需但建议存在
missing = [r for r in required if r not in cols]
if len(missing) > 0:
    print("Detected columns in the file:")
    for i,c in enumerate(df_raw.columns):
        print(f"{i:02d}: {c!r}")
    raise RuntimeError(f"Cannot find required columns {missing} in {data_path}. Detected map so far: {cols}")

# 重新命名 DataFrame 列到规范名
df = df_raw.rename(columns={ cols[k]: k for k in cols })
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# ensure numeric and fill
for c in ['exog_temp','exog_wind','OT']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.interpolate(limit=5).ffill().bfill()
df['OT_prev'] = df['OT'].shift(1)
df = df.dropna().reset_index(drop=True)

# ----------------- Re-create SeqDataset (same as in training) -----------------
from torch.utils.data import Dataset, DataLoader

feat_cols = ['exog_temp','exog_wind','OT_prev']
# if exog_wind missing, adjust
feat_cols = [c for c in feat_cols if c in df.columns]

class SeqDatasetForEmb(Dataset):
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols):
        self.df = df_scaled
        self.start = start_idx
        self.end = end_idx
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        self.n = max(0, (self.end - self.start + 1) - self.seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i0 = self.start + idx
        seq = self.df.iloc[i0: i0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = i0 + self.seq_len
        y = self.df.iloc[label_idx]['OT'].astype(np.float32)
        return seq, y, int(label_idx)

# ----------------- Load scaler & produce scaled df (like training) -----------------
if SCALER_PATH.exists():
    scaler_inputs = joblib.load(SCALER_PATH)
    df_scaled = df.copy()
    df_scaled[feat_cols] = scaler_inputs.transform(df[feat_cols].values)
else:
    # fallback: standard scale using training-like split (use full for safety)
    from sklearn.preprocessing import StandardScaler
    scaler_inputs = StandardScaler().fit(df[feat_cols].values)
    df_scaled = df.copy()
    df_scaled[feat_cols] = scaler_inputs.transform(df[feat_cols].values)
    joblib.dump(scaler_inputs, SCALER_PATH)
    print(f"[WARN] scaler_inputs not found, fit on full df and saved to {SCALER_PATH}")

# ----------------- Build test loader (chronological split same as training logic) -----------------
n = len(df_scaled)
test_size = int(n * 0.20)   # default TEST_RATIO used in training script
val_size  = int(n * 0.10)
train_size = n - test_size - val_size
train_end = train_size - 1
val_end = train_size + val_size - 1
test_start = train_size + val_size
test_end = n - 1

test_ds = SeqDatasetForEmb(df_scaled, test_start, test_end, SEQ_LEN, feat_cols)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Test samples:", len(test_ds))

# ----------------- Recreate model class (must match saved model) -----------------
import torch.nn as nn

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feat_dim=3, cnn_channels=64, cnn_kernel=3, lstm_hid=128,
                 d_model=128, nhead=4, num_transformer_layers=0, dropout_rate=0.0,
                 heteroscedastic=False):
        super().__init__()
        self.hetero = heteroscedastic
        self.conv1 = nn.Conv1d(in_channels=feat_dim, out_channels=cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hid,
                            num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        if lstm_hid != d_model:
            self.proj_to_d = nn.Linear(lstm_hid, d_model)
        else:
            self.proj_to_d = None
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_rate)
        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                       dim_feedforward=d_model*2,
                                                       batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=num_transformer_layers)
        else:
            self.transformer_encoder = None
        final_out_dim = 2 if USE_HETEROSCEDASTIC else 1
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, final_out_dim)
        )
    def forward(self, x):
        c = x.permute(0,2,1)
        c = self.act(self.conv1(c))
        c = self.act(self.conv2(c))
        c = self.dropout_cnn(c)
        c = c.permute(0,2,1)
        lstm_out, _ = self.lstm(c)
        lstm_out = self.dropout_lstm(lstm_out)
        if self.proj_to_d is not None:
            tr_in = self.proj_to_d(lstm_out)
        else:
            tr_in = lstm_out
        attn_out, _ = self.mha(tr_in, tr_in, tr_in, need_weights=False)
        attn_out = self.dropout_attn(attn_out)
        tr_out = self.transformer_encoder(attn_out) if self.transformer_encoder else attn_out
        last = tr_out[:, -1, :]         # embedding we want
        out = self.fc(last)
        if self.hetero:
            mu = out[:,0]; logvar = out[:,1].clamp(-10,10)
            return mu, logvar
        else:
            return out.squeeze(1)

# ----------------- Load model weights -----------------
model_path = GLOBAL_MODEL_PATH if GLOBAL_MODEL_PATH.exists() else (FALLBACK_MODEL if FALLBACK_MODEL.exists() else None)
if model_path is None:
    raise FileNotFoundError(f"No global model found at {GLOBAL_MODEL_PATH} or fallback {FALLBACK_MODEL}. Please provide a trained global model.")

print("Loading model from:", model_path)
# create model with same hyperparams you used for training (adjust if needed)
feat_dim = len(feat_cols)
model = CNN_LSTM_Attention(feat_dim=feat_dim, cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL,
                           lstm_hid=LSTM_HID, d_model=TRANS_DMODEL, nhead=NUM_HEADS,
                           num_transformer_layers=NUM_TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE,
                           heteroscedastic=USE_HETEROSCEDASTIC)
state = torch.load(model_path, map_location=DEVICE)
# if state is a dict of tensors (state_dict saved), load it; else if saved as raw state, try both
if isinstance(state, dict) and any(k.startswith('conv1') or k.startswith('fc') for k in state.keys()):
    model.load_state_dict(state)
else:
    # maybe the saved file is a raw dict with 'model' key or similar
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: could not directly load state_dict; attempting direct assignment may fail.")
            raise e

model.to(DEVICE).eval()

# ----------------- Hook to capture 'last' (fc input) embeddings -----------------
embeddings = []
meta_label_idxs = []

# shared container used by hook
_hook_buffer = {}
def make_hook(buf):
    def hook(module, input, output):
        # input is a tuple; input[0] is 'last' shape (B, d_model)
        arr = input[0].detach().cpu().numpy()
        # append arr as-is
        buf.setdefault('arrs', []).append(arr)
    return hook

buf = {}
hook_handle = model.fc.register_forward_hook(make_hook(buf))

# forward through test_loader and collect embeddings and label idxs
with torch.no_grad():
    for xb, yb, idxs in test_loader:
        buf.clear()
        xb = xb.to(DEVICE)
        # forward: triggers hook and stores input to fc in buf['arrs']
        _out = model(xb)
        arrs = buf.get('arrs', [])
        if len(arrs) == 0:
            # unexpected: hook didn't capture; skip
            print("Warning: no activation captured for a batch; skipping")
            continue
        # arrs[0] is (B, d_model) for this batch
        batch_feats = arrs[0]
        embeddings.append(batch_feats)
        meta_label_idxs.extend([int(i) for i in idxs.numpy().tolist()])

# remove hook
hook_handle.remove()

if len(embeddings) == 0:
    raise RuntimeError("No embeddings were extracted. Check hook and model architecture.")

embeddings = np.vstack(embeddings)  # shape (N_test, d)
print("Embeddings shape:", embeddings.shape)
np.save(EMB_NPY, embeddings)
pd.DataFrame({'label_idx': meta_label_idxs}).to_csv(META_CSV, index=False)
print("Saved embeddings to:", EMB_NPY, "and meta to:", META_CSV)

# ----------------- KMeans(2) on embeddings -----------------
kmeans_emb = KMeans(n_clusters=2, random_state=42, n_init=20).fit(embeddings)
cluster_labels = kmeans_emb.labels_

# ----------------- Prepare scatter data: x = OT_true, y = exog_temp (use original df) ---------------
# meta_label_idxs maps each embedding to a label_idx (global index of true label in df)
label_idxs = np.array(meta_label_idxs, dtype=int)
ot_true_vals = df.loc[label_idxs, 'OT'].values
temp_vals = df.loc[label_idxs, 'exog_temp'].values

# build DataFrame for plotting & analysis
plot_df = pd.DataFrame({
    'label_idx': label_idxs,
    'OT_true': ot_true_vals,
    'exog_temp': temp_vals,
    'emb_cluster': cluster_labels
})

# optionally sort by cluster for plotting (so colors are grouped)
plot_df = plot_df.sort_values('emb_cluster').reset_index(drop=True)

# ----------------- Scatter plot -----------------
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", n_colors=2)
for c in np.unique(cluster_labels):
    sub = plot_df[plot_df['emb_cluster']==c]
    plt.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.8, label=f'cluster_{c}', color=palette[int(c)])
plt.xlabel("OT (true power)")
plt.ylabel("Temperature (exog_temp)")
plt.title("Embeddings-based clustering (KMeans=3) — points colored by cluster")
plt.legend()
plt.tight_layout()
plt.savefig(SCATTER_PNG, dpi=300, bbox_inches='tight')
plt.close()
print("Saved scatter to:", SCATTER_PNG)

# ----------------- Save cluster labels back to CSV for inspection -----------------
plot_df.to_csv(OUT_DIR / "emb_cluster_vs_ot_temp.csv", index=False)
print("Saved emb_cluster_vs_ot_temp.csv to:", OUT_DIR)
