"""ゼロから学習する小さなチャットAI（文字単位GPT）
使い方:
  python chat_gpt.py train   # data.txt で学習して model.pt / vocab.json を保存
  python chat_gpt.py chat    # 学習済みモデルとチャット
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# CPU(Codespaces)向けの軽い設定。GPUなら大きくしてOK
block_size, n_embd, n_head, n_layer, dropout = 128, 128, 4, 2, 0.1
batch_size, max_iters, lr = 32, int(os.environ.get('ITERS', 1500)), 5e-4


class Block(nn.Module):
    def __init__(s, n_embd, n_head):
        super().__init__()
        s.ln1 = nn.LayerNorm(n_embd)
        s.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        s.ln2 = nn.LayerNorm(n_embd)
        s.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                             nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(s, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = s.ln1(x)
        a, _ = s.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + s.ff(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        s.vocab_size = vocab_size
        s.tok = nn.Embedding(vocab_size, n_embd)
        s.pos = nn.Embedding(block_size, n_embd)
        s.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        s.ln = nn.LayerNorm(n_embd)
        s.head = nn.Linear(n_embd, vocab_size)

    def forward(s, idx, targets=None):
        T = idx.shape[1]
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        logits = s.head(s.ln(s.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, s.vocab_size), targets.view(-1))
        return logits, loss


def train():
    text = open('data.txt', encoding='utf-8').read()
    if len(text) < 20000:  # 少量データは繰り返して学習量を確保
        text = text * (20000 // max(len(text), 1) + 1)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * 0.9)
    tr, va = data[:n], data[n:]
    print('文字数', len(text), '語彙数', len(chars), 'device', device)

    model = GPT(len(chars), n_embd, n_head, n_layer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    for it in range(max_iters + 1):
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*batch(va))[1].item() for _ in range(10)) / 10
            model.train()
            print(f'iter {it}  val loss {vl:.3f}', flush=True)
        _, loss = model(*batch(tr))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), 'model.pt')
    json.dump({'chars': chars}, open('vocab.json', 'w'), ensure_ascii=False)
    print('保存しました: model.pt / vocab.json')


@torch.no_grad()
def reply(model, stoi, itos, msg, temperature=0.7, top_k=20, max_new=120):
    model.eval()
    prompt = f'ユーザー:{msg}\nAI:'
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    out = ''
    for _ in range(max_new):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, model.vocab_size))
        logits[logits < v[:, [-1]]] = -float('inf')
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        ch = itos[nxt.item()]
        if ch == '\n':
            break
        out += ch
    return out


def chat():
    chars = json.load(open('vocab.json', encoding='utf-8'))['chars']
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = GPT(len(chars), n_embd, n_head, n_layer).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    print('チャット開始（終了は Ctrl+C か「quit」）')
    while True:
        msg = input('あなた: ').strip()
        if msg in ('quit', 'exit'):
            break
        print('AI:', reply(model, stoi, itos, msg))


if __name__ == '__main__':
    {'train': train, 'chat': chat}.get(sys.argv[1] if len(sys.argv) > 1 else '', lambda: print(__doc__))()
