import sys
from unittest.mock import MagicMock
sys.modules['albumentations'] = MagicMock()
sys.modules['albumentations.pytorch'] = MagicMock()
sys.modules['albumentations.core'] = MagicMock()
sys.modules['albumentations.core.transforms_interface'] = MagicMock()
sys.modules['imgaug'] = MagicMock()
sys.modules['imgaug.augmenters'] = MagicMock()

import torch, cv2, numpy as np, argparse, pickle, os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd

sys.path.append('Mammo-CLIP/src/codebase')
from Classifiers.models.breast_clip_classifier import BreastClipClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

# ── Charger encodeur ───────────────────────────────────────
ckpt_pre = torch.load('models_export/Pre-trained-checkpoints/b5-model-best-epoch-7.tar',
                       map_location='cpu', weights_only=False)
args      = argparse.Namespace()
args.arch = 'upmc_breast_clip_det_b5_period_n_ft'
encoder   = BreastClipClassifier(args=args, ckpt=ckpt_pre, n_class=1).to(device).eval()
print('Encodeur charge !')

# ── Charger CSV VinDr ──────────────────────────────────────
csv_path   = 'Mammo-CLIP/src/codebase/data_csv/vindr_detection_v1_folds.csv'
img_base   = 'vindr_images/'  # ← chemin vers tes images VinDr en local
df         = pd.read_csv(csv_path)
df_train   = df[df['split'] == 'training'].drop_duplicates('image_id')
df_test    = df[df['split'] == 'test'].drop_duplicates('image_id')
print(f'Train : {len(df_train)}  Test : {len(df_test)}')

# ── Extraire features ──────────────────────────────────────
def extract_features(df_split):
    feats, labels = [], []
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        img_path = f'{img_base}/{row["patient_id"]}/{row["image_id"]}'
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (456, 760))
        img = (img.astype(np.float32) / 255.0 - 0.3089279) / 0.25053555
        img = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)
        with torch.no_grad():
            feat, _ = encoder.image_encoder({'image': img})
        feats.append(feat.cpu().numpy()[0])
        labels.append(int(row['Mass']))
    return np.array(feats), np.array(labels)

print('Extraction features train...')
X_train, y_train = extract_features(df_train)
print('Extraction features test...')
X_test,  y_test  = extract_features(df_test)

# ── Entrainer Linear Probe ─────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                          penalty='l2', random_state=42)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, probs)
print(f'\nLinear Probe AUC (2048 features) : {auc:.4f}')

# ── Sauvegarder ────────────────────────────────────────────
lp_new = {'Mass': {'classifier': clf, 'scaler': scaler, 'auc': auc}}
with open('linear_probe_models_2048.pkl', 'wb') as f:
    pickle.dump(lp_new, f)
print('Sauvegarde : linear_probe_models_2048.pkl ✅')
