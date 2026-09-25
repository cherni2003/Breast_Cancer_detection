"""
==============================================================
  Mammo-CLIP — Inférence + GradCAM + RetinaNet
  Extension de ton inference.py existant
==============================================================

PRÉREQUIS (une seule fois) :
    pip install torch torchvision efficientnet-pytorch pillow matplotlib numpy opencv-python

USAGE :
    python inference_retinanet.py --image mammonor.PNG

PIPELINE :
    PNG
     ├─→ BreastClipClassifier  →  Score cancer (ton modèle AUC 0.796)
     ├─→ GradCAM               →  Heatmap activation
     └─→ RetinaNet (B5 CLIP)   →  Bounding boxes détection masse

STRUCTURE DES FICHIERS ATTENDUS (identique à ton projet) :
    models_export/
        Pre-trained-checkpoints/b5-model-best-epoch-7.tar
        best_model_auc0796_recall0853.pth
    Mammo-CLIP/src/codebase/
        Classifiers/models/breast_clip_classifier.py
    mammonor.PNG   ← (ou toute autre image PNG)
==============================================================
"""

# ── Mocks albumentations (identique à ton inference.py) ────
import sys
from unittest.mock import MagicMock
sys.modules['albumentations']                           = MagicMock()
sys.modules['albumentations.pytorch']                   = MagicMock()
sys.modules['albumentations.core']                      = MagicMock()
sys.modules['albumentations.core.transforms_interface'] = MagicMock()
sys.modules['imgaug']                                   = MagicMock()
sys.modules['imgaug.augmenters']                        = MagicMock()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.append('Mammo-CLIP/src/codebase')
from Classifiers.models.breast_clip_classifier import BreastClipClassifier


# ══════════════════════════════════════════════════════════════
#  CONFIG GLOBALE
# ══════════════════════════════════════════════════════════════

CKPT_PRETRAINED = 'models_export/Pre-trained-checkpoints/b5-model-best-epoch-7.tar'
CKPT_FINETUNED  = 'models_export/best_model_auc0796_recall0853.pth'
ARCH            = 'upmc_breast_clip_det_b5_period_n_ft'

# Normalisation Mammo-CLIP (identique à ton script)
MAMMO_MEAN = 0.3089279
MAMMO_STD  = 0.25053555

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device : {DEVICE}')


# ══════════════════════════════════════════════════════════════
#  1. Chargement du modèle (identique à ton inference.py)
# ══════════════════════════════════════════════════════════════

def load_classifier():
    """Charge BreastClipClassifier avec ton checkpoint fine-tuné."""
    print('[MODÈLE] Chargement BreastClipClassifier B5...')
    ckpt_pre = torch.load(CKPT_PRETRAINED, map_location='cpu', weights_only=False)
    ckpt_ft  = torch.load(CKPT_FINETUNED,  map_location='cpu', weights_only=False)

    args      = argparse.Namespace()
    args.arch = ARCH
    model     = BreastClipClassifier(args=args, ckpt=ckpt_pre, n_class=1)
    model.load_state_dict(ckpt_ft['model'], strict=False)
    model     = model.to(DEVICE).eval()
    print('[MODÈLE] BreastClipClassifier chargé ✓  (AUC 0.796)')
    return model


# ══════════════════════════════════════════════════════════════
#  2. Backbone EfficientNet-B5 + FPN pour RetinaNet
#     On réutilise l'image_encoder du modèle déjà chargé !
# ══════════════════════════════════════════════════════════════

def _find_effnet(image_encoder):
    """
    Cherche le module EfficientNet réel dans l'image_encoder
    de BreastClipClassifier (peu importe la profondeur d'imbrication).
    """
    for _, module in image_encoder.named_modules():
        if hasattr(module, '_blocks') and hasattr(module, '_conv_head'):
            return module
    return None


def _probe_block_channels(effnet_module, device, img_size=512):
    """
    Fait passer un tensor factice dans chaque bloc d'EfficientNet
    pour mesurer le nombre de canaux de sortie réel.
    Retourne une liste [(idx, n_channels), ...] triée par idx.
    """
    print('[FPN-PROBE] Détection automatique des canaux par bloc...')
    channels = []
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    acts = {}
    handles = []
    for i, block in enumerate(effnet_module._blocks):
        def hook(idx):
            def _h(m, inp, out):
                acts[idx] = out.shape[1]   # nb canaux
            return _h
        handles.append(block.register_forward_hook(hook(i)))

    with torch.no_grad():
        effnet_module.extract_features(dummy)

    for h in handles:
        h.remove()

    channels = sorted(acts.items())   # [(0, ch0), (1, ch1), ...]
    n_blocks  = len(channels)

    # Afficher un résumé compact
    unique_ch = []
    for i, ch in channels:
        if not unique_ch or unique_ch[-1][1] != ch:
            unique_ch.append((i, ch))
    print(f'[FPN-PROBE] {n_blocks} blocs — transitions de canaux :')
    for i, ch in unique_ch:
        print(f'            bloc[{i:2d}] → {ch} canaux')

    return channels, n_blocks


def _pick_fpn_indices(channels_list):
    """
    Choisit automatiquement 3 indices de blocs bien répartis
    correspondant aux 3 niveaux FPN (C3, C4, C5).

    Stratégie :
      - C3 ≈ 25% des blocs  (features fines)
      - C4 ≈ 60% des blocs  (features moyennes)
      - C5 = dernier bloc    (features profondes)
    """
    n = len(channels_list)
    i_c3 = int(n * 0.25)
    i_c4 = int(n * 0.60)
    i_c5 = n - 1

    ch_c3 = channels_list[i_c3][1]
    ch_c4 = channels_list[i_c4][1]
    ch_c5 = channels_list[i_c5][1]

    print(f'[FPN-PROBE] Niveaux FPN choisis :')
    print(f'            C3 → bloc[{i_c3}]  {ch_c3} canaux')
    print(f'            C4 → bloc[{i_c4}]  {ch_c4} canaux')
    print(f'            C5 → bloc[{i_c5}]  {ch_c5} canaux')

    return (i_c3, ch_c3), (i_c4, ch_c4), (i_c5, ch_c5)


class EffNetB5FPN(nn.Module):
    """
    Wrapper autour de l'image_encoder BreastClipClassifier.
    Détecte automatiquement les canaux réels → plus d'erreur de dimensions.

    Les indices et canaux FPN sont probes à l'init avec un tensor factice,
    donc ça marche quelle que soit la version d'efficientnet-pytorch.
    """
    FPN_CH = 256

    def __init__(self, image_encoder):
        super().__init__()
        self.effnet_enc = image_encoder
        self.out_channels = self.FPN_CH

        # Trouver le module EfficientNet réel
        self.effnet_module = _find_effnet(image_encoder)
        if self.effnet_module is None:
            raise RuntimeError(
                "[FPN] Impossible de trouver EfficientNet dans image_encoder.\n"
                "Vérifie que image_encoder contient bien un module avec _blocks."
            )

        # Détecter les canaux réels de chaque bloc
        dev = next(image_encoder.parameters()).device
        channels_list, _ = _probe_block_channels(self.effnet_module, dev)

        # Choisir les 3 niveaux FPN automatiquement
        (self.i_c3, c3_ch), \
        (self.i_c4, c4_ch), \
        (self.i_c5, c5_ch) = _pick_fpn_indices(channels_list)

        # Projections 1×1 avec les VRAIS nombres de canaux
        self.proj_c3 = nn.Conv2d(c3_ch, self.FPN_CH, 1)
        self.proj_c4 = nn.Conv2d(c4_ch, self.FPN_CH, 1)
        self.proj_c5 = nn.Conv2d(c5_ch, self.FPN_CH, 1)

        # Lissage top-down
        self.smooth_p4 = nn.Conv2d(self.FPN_CH, self.FPN_CH, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(self.FPN_CH, self.FPN_CH, 3, padding=1)

        # P6 et P7 pour les grandes masses
        self.p6 = nn.Conv2d(self.FPN_CH, self.FPN_CH, 3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.FPN_CH, self.FPN_CH, 3, stride=2, padding=1)
        )

    def forward(self, x):
        acts = {}

        def hook(name):
            def _h(m, inp, out): acts[name] = out
            return _h

        blocks = self.effnet_module._blocks
        h1 = blocks[self.i_c3].register_forward_hook(hook("c3"))
        h2 = blocks[self.i_c4].register_forward_hook(hook("c4"))
        h3 = blocks[self.i_c5].register_forward_hook(hook("c5"))

        self.effnet_module.extract_features(x)

        h1.remove(); h2.remove(); h3.remove()

        c3 = self.proj_c3(acts["c3"])
        c4 = self.proj_c4(acts["c4"])
        c5 = self.proj_c5(acts["c5"])

        # Top-down FPN
        p5 = c5
        p4 = self.smooth_p4(
            c4 + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        )
        p3 = self.smooth_p3(
            c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        )
        p6 = self.p6(p5)
        p7 = self.p7(p6)

        return {"0": p3, "1": p4, "2": p5, "3": p6, "4": p7}


def build_retinanet(classifier_model, num_classes=1):
    """
    Construit RetinaNet en réutilisant l'image_encoder du classifieur.
    Les poids Mammo-CLIP sont déjà chargés — aucun rechargement nécessaire.
    """
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    print('[RETINANET] Construction avec backbone Mammo-CLIP déjà chargé...')

    backbone = EffNetB5FPN(classifier_model.image_encoder)

    base = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    )
    base.backbone = backbone

    # Adapter la tête au nombre de classes
    in_ch       = base.head.classification_head.conv[0][0].in_channels
    num_anchors = base.head.classification_head.num_anchors
    base.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_ch,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    base = base.to(DEVICE).eval()
    n = sum(p.numel() for p in base.parameters())
    print(f'[RETINANET] Prêt ✓  ({n:,} paramètres)')
    return base


# ══════════════════════════════════════════════════════════════
#  3. GradCAM (identique à ton inference.py, légèrement nettoyé)
# ══════════════════════════════════════════════════════════════

def gradcam_heatmap_masked(img_path, model, img_size=512):
    """GradCAM avec masque tissu mammaire — identique à ton script."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'[ERREUR] Image non trouvée : {img_path}')
        return None

    img_display = cv2.resize(img, (img_size, img_size))

    # Masque tissu mammaire
    _, breast_mask = cv2.threshold(img_display, 15, 255, cv2.THRESH_BINARY)
    kernel         = np.ones((20, 20), np.uint8)
    breast_mask    = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
    breast_mask    = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN,  kernel)
    contours, _    = cv2.findContours(breast_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        breast_mask = np.zeros_like(breast_mask)
        cv2.fillPoly(breast_mask,
                     [max(contours, key=cv2.contourArea)], 255)

    # Préparer tensor pour le modèle (normalisation Mammo-CLIP)
    img_r    = cv2.resize(img, (456, 760))
    img_norm = (img_r.astype(np.float32) / 255.0 - MAMMO_MEAN) / MAMMO_STD
    img_t    = (torch.tensor(img_norm)
                .unsqueeze(0).repeat(3, 1, 1)
                .unsqueeze(0).to(DEVICE))
    img_t.requires_grad_(True)

    # Hooks GradCAM
    activations, gradients = {}, {}
    def fwd_hook(m, inp, out): activations['f'] = out
    def bwd_hook(m, gi,  go):  gradients['f']   = go[0]

    h_f = model.image_encoder._conv_head.register_forward_hook(fwd_hook)
    h_b = model.image_encoder._conv_head.register_full_backward_hook(bwd_hook)

    feat, _ = model.image_encoder({'image': img_t})
    prob     = torch.sigmoid(model.classifier(feat))
    prob.backward()
    h_f.remove(); h_b.remove()

    # Générer heatmap
    grads = gradients['f'].mean(dim=[2, 3], keepdim=True)
    cam   = F.relu((grads * activations['f']).sum(dim=1, keepdim=True))
    cam   = F.interpolate(cam, size=(img_size, img_size),
                          mode='bilinear', align_corners=False)
    cam   = cam.squeeze().detach().cpu().numpy()

    # Masque sein
    cam[breast_mask == 0] = 0
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Overlay couleur
    heatmap     = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_rgb     = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    mask_3d     = np.stack([breast_mask / 255] * 3, axis=2)
    overlay     = (img_rgb * (1 - mask_3d * 0.5)
                   + heatmap_rgb * mask_3d * 0.5).astype(np.uint8)

    # BBox zone chaude GradCAM
    hot_mask    = ((cam > 0.5) & (breast_mask > 0)).astype(np.uint8)
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    bbox_cam = None
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c_max) > 100:
            x, y, w, h = cv2.boundingRect(c_max)
            pad      = 15
            bbox_cam = [max(0, x-pad), max(0, y-pad),
                        min(img_size, x+w+pad), min(img_size, y+h+pad)]

    return img_display, cam, overlay, prob.item(), breast_mask, bbox_cam


# ══════════════════════════════════════════════════════════════
#  4. Inférence RetinaNet sur l'image PNG
# ══════════════════════════════════════════════════════════════

def retinanet_detect(img_path, retinanet, img_size=512,
                     score_threshold=0.10):
    """
    Prépare l'image avec la normalisation ImageNet (standard RetinaNet)
    et lance la détection.
    Retourne (boxes, scores, labels).
    """
    pil = Image.open(img_path).convert("RGB")

    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    tensor = tfm(pil).to(DEVICE)

    retinanet.eval()
    with torch.no_grad():
        preds = retinanet([tensor])

    pred   = preds[0]
    boxes  = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    keep   = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    print(f'[RETINANET] Détections (seuil {score_threshold}) : {len(boxes)}')
    for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = b
        print(f'  #{i+1}  score={s:.3f}  '
              f'box=[{x1:.0f},{y1:.0f}→{x2:.0f},{y2:.0f}]  '
              f'({x2-x1:.0f}×{y2-y1:.0f}px)')

    return boxes, scores, labels


# ══════════════════════════════════════════════════════════════
#  5. Visualisation combinée (4 panneaux)
# ══════════════════════════════════════════════════════════════

RETINA_COLORS = ['#FF4444', '#FF8C00', '#00CFFF', '#00FF88', '#FFD700']
RETINA_LABELS = {1: 'Mass', 2: 'Calcification', 3: 'Asymmetry'}


def visualize_combined(img_path, img_display, cam, overlay,
                        prob, breast_mask, bbox_cam,
                        boxes_ret, scores_ret, labels_ret,
                        img_size, output_path, score_threshold):
    """
    4 panneaux :
      [0] Image originale
      [1] GradCAM heatmap masquée  (ton code d'origine)
      [2] Overlay GradCAM + bbox GradCAM  (ton code d'origine)
      [3] RetinaNet bounding boxes  ← NOUVEAU
    """
    pred  = 'MASSE ⚠️'  if prob >= 0.5 else 'Normal ✅'
    color = 'red'       if prob >= 0.5 else 'green'

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='#111111')
    fig.suptitle(
        f'Mammo-CLIP — {os.path.basename(img_path)}\n'
        f'Prédiction : {pred}   Probabilité : {prob*100:.1f}%',
        fontsize=13, fontweight='bold', color=color
    )
    for ax in axes:
        ax.set_facecolor('#111111')
        ax.axis('off')

    # ── Panneau 0 : image originale ───────────────────────────
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Image originale', color='white', fontsize=11)

    # ── Panneau 1 : GradCAM heatmap ───────────────────────────
    cam_display = cam.copy().astype(float)
    cam_display[breast_mask == 0] = np.nan
    axes[1].imshow(img_display, cmap='gray')
    im = axes[1].imshow(cam_display, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title('GradCAM heatmap\n🔴 Masse  🔵 Normal',
                      color='white', fontsize=11)
    if bbox_cam:
        x1c, y1c, x2c, y2c = bbox_cam
        axes[1].add_patch(patches.Rectangle(
            (x1c, y1c), x2c-x1c, y2c-y1c,
            lw=2, edgecolor='white', facecolor='none', linestyle='--'
        ))
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label='Activation')

    # ── Panneau 2 : Overlay GradCAM ───────────────────────────
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay GradCAM + bbox suspecte',
                      color='white', fontsize=11)
    if bbox_cam:
        x1c, y1c, x2c, y2c = bbox_cam
        axes[2].add_patch(patches.Rectangle(
            (x1c, y1c), x2c-x1c, y2c-y1c,
            lw=2, edgecolor='red', facecolor='none'
        ))

    # ── Panneau 3 : RetinaNet ─────────────────────────────────
    axes[3].imshow(img_display, cmap='gray')
    n_ret = len(boxes_ret)
    axes[3].set_title(
        f'RetinaNet — {n_ret} détection{"s" if n_ret != 1 else ""}'
        f'  (seuil {score_threshold:.2f})',
        color='white', fontsize=11
    )

    if n_ret == 0:
        axes[3].text(
            img_size * 0.5, img_size * 0.5,
            'Aucune détection\n(seuil actuel)\n\n'
            '→ Baisse --score_threshold\n'
            '→ Fine-tune train_detector.py',
            ha='center', va='center', fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.6',
                      facecolor='#1a1a55', edgecolor='#5555FF', lw=1.5)
        )
    else:
        for i, (box, score, lbl) in enumerate(
                zip(boxes_ret, scores_ret, labels_ret)):
            x1, y1, x2, y2 = box
            c    = RETINA_COLORS[i % len(RETINA_COLORS)]
            name = RETINA_LABELS.get(int(lbl), f'Classe {int(lbl)}')
            axes[3].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                lw=2.5, edgecolor=c, facecolor='none'
            ))
            axes[3].text(
                x1 + 3, max(y1 - 10, 3),
                f' {name}  {score:.2f} ',
                color='white', fontsize=9, fontweight='bold', va='top',
                bbox=dict(facecolor=c, edgecolor='none',
                          alpha=0.85, boxstyle='round,pad=0.3')
            )
            cx, cy = (x1+x2)/2, (y1+y2)/2
            axes[3].plot(cx, cy, '+', color=c, ms=10, mew=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#111111')
    plt.close()
    print(f'[OK] Résultat sauvegardé → {output_path}')


# ══════════════════════════════════════════════════════════════
#  6. Résumé terminal
# ══════════════════════════════════════════════════════════════

def print_summary(img_path, prob, bbox_cam,
                  boxes_ret, scores_ret, output_path):
    pred = 'MASSE ⚠️' if prob >= 0.5 else 'Normal ✅'
    print(f'\n{"="*50}')
    print(f'  RÉSUMÉ — {os.path.basename(img_path)}')
    print(f'{"="*50}')
    print(f'  Classifieur  : {pred}  ({prob*100:.1f}%)')
    print(f'  GradCAM bbox : {bbox_cam}')
    print(f'  RetinaNet    : {len(boxes_ret)} détection(s)')
    if len(scores_ret):
        tops = sorted(scores_ret, reverse=True)[:3]
        print(f'  Top scores   : {[f"{s:.3f}" for s in tops]}')

    # Décision de confiance combinée
    print(f'\n  ── Confiance combinée ──────────────────────')
    high_prob   = prob >= 0.5
    has_ret     = len(boxes_ret) > 0
    has_gradcam = bbox_cam is not None

    if high_prob and has_ret and has_gradcam:
        conf = '🔴 HAUTE — Classifieur + GradCAM + RetinaNet convergent'
    elif high_prob and (has_ret or has_gradcam):
        conf = '🟠 MOYENNE — Classifieur positif + 1 méthode de localisation'
    elif high_prob:
        conf = '🟡 FAIBLE — Classifieur positif mais pas de localisation'
    else:
        conf = '🟢 NORMALE — Classifieur négatif'

    print(f'  {conf}')
    print(f'{"="*50}')
    print(f'  PNG sauvegardé : {output_path}')
    print(f'{"="*50}')


# ══════════════════════════════════════════════════════════════
#  7. Fonction principale predict()
# ══════════════════════════════════════════════════════════════

def predict(img_path, classifier, retinanet,
            img_size=512, score_threshold=0.10,
            output_dir='.'):
    """
    Pipeline complet :
      1) GradCAM via BreastClipClassifier
      2) Détection RetinaNet
      3) Visualisation 4 panneaux
      4) Résumé terminal
    """
    print(f'\n[PREDICT] Image : {img_path}')

    # GradCAM + score classifieur
    result = gradcam_heatmap_masked(img_path, classifier, img_size)
    if result is None:
        return
    img_display, cam, overlay, prob, breast_mask, bbox_cam = result
    print(f'[CLASSIFIEUR] Probabilité masse : {prob*100:.1f}%')

    # RetinaNet
    boxes_ret, scores_ret, labels_ret = retinanet_detect(
        img_path, retinanet, img_size, score_threshold
    )

    # Visualisation
    stem       = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir,
                               f'gradcam_retinanet_{stem}.png')
    os.makedirs(output_dir, exist_ok=True)

    visualize_combined(
        img_path, img_display, cam, overlay,
        prob, breast_mask, bbox_cam,
        boxes_ret, scores_ret, labels_ret,
        img_size, output_path, score_threshold
    )

    # Résumé
    print_summary(img_path, prob, bbox_cam,
                  boxes_ret, scores_ret, output_path)

    return {
        'prob'       : prob,
        'bbox_gradcam': bbox_cam,
        'boxes_ret'  : boxes_ret,
        'scores_ret' : scores_ret,
        'output'     : output_path,
    }


# ══════════════════════════════════════════════════════════════
#  8. CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Mammo-CLIP — GradCAM + RetinaNet'
    )
    p.add_argument('--image',           type=str,   default='mammonor.PNG',
                   help='Image PNG à analyser')
    p.add_argument('--score_threshold', type=float, default=0.10,
                   help='Seuil RetinaNet (défaut : 0.10)')
    p.add_argument('--img_size',        type=int,   default=512,
                   help='Taille resize (défaut : 512)')
    p.add_argument('--output_dir',      type=str,   default='.',
                   help='Dossier de sortie (défaut : .)')
    p.add_argument('--retina_ckpt',     type=str,   default=None,
                   help='Checkpoint RetinaNet fine-tuné (optionnel)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Charger les modèles une seule fois
    classifier = load_classifier()
    retinanet  = build_retinanet(classifier, num_classes=1)

    # Charger un checkpoint RetinaNet fine-tuné si disponible
    if args.retina_ckpt and os.path.exists(args.retina_ckpt):
        print(f'[RETINANET] Chargement checkpoint : {args.retina_ckpt}')
        ckpt = torch.load(args.retina_ckpt, map_location='cpu')
        sd   = ckpt.get('model', ckpt.get('state_dict', ckpt))
        retinanet.load_state_dict(sd, strict=False)
        print('[RETINANET] Checkpoint chargé ✓')

    # Lancer la prédiction
    predict(
        img_path        = args.image,
        classifier      = classifier,
        retinanet       = retinanet,
        img_size        = args.img_size,
        score_threshold = args.score_threshold,
        output_dir      = args.output_dir,
    )