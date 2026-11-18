"""学習済み model.pt をブラウザ用の model.json に変換する。
使い方: python export_web.py
"""
import json, torch

# chat_gpt.py と同じ設定にすること
n_embd, n_head, n_layer, block_size = 128, 4, 2, 128

chars = json.load(open('vocab.json', encoding='utf-8'))['chars']
sd = torch.load('model.pt', map_location='cpu')
weights = {}
for k, v in sd.items():
    weights[k] = {'shape': list(v.shape),
                  'data': [round(x, 5) for x in v.flatten().tolist()]}
out = {'config': {'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer,
                  'block_size': block_size},
       'chars': chars, 'weights': weights}
json.dump(out, open('model.json', 'w', encoding='utf-8'), ensure_ascii=False,
          separators=(',', ':'))
print('model.json を作りました')
