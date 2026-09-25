# test_path.py — mets ce fichier dans backend/
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

codebase = os.path.join(ROOT, 'Mammo-CLIP', 'src', 'codebase')
print(f'ROOT     : {ROOT}')
print(f'codebase : {codebase}')
print(f'existe   : {os.path.exists(codebase)}')
print(f'contenu  : {os.listdir(codebase) if os.path.exists(codebase) else "INTROUVABLE"}')