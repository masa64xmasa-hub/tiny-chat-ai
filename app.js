// 小さなGPTをブラウザ(JS)だけで動かす。PyTorchの chat_gpt.py と同じ計算。
function erf(x) { // Abramowitz-Stegun 7.1.26 より高精度な近似
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function layerNorm(x, T, E, w, b) {
  const out = new Float32Array(T * E);
  for (let t = 0; t < T; t++) {
    let m = 0; for (let i = 0; i < E; i++) m += x[t * E + i]; m /= E;
    let v = 0; for (let i = 0; i < E; i++) { const d = x[t * E + i] - m; v += d * d; } v /= E;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < E; i++) out[t * E + i] = (x[t * E + i] - m) * inv * w[i] + b[i];
  }
  return out;
}
function linear(x, T, I, W, b, O) {
  const out = new Float32Array(T * O);
  for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) {
    let s = b ? b[o] : 0;
    for (let i = 0; i < I; i++) s += x[t * I + i] * W[o * I + i];
    out[t * O + o] = s;
  }
  return out;
}
function loadModel(json) {
  const w = {};
  for (const k in json.weights) w[k] = Float32Array.from(json.weights[k].data);
  const stoi = {}; json.chars.forEach((c, i) => stoi[c] = i);
  return { cfg: json.config, chars: json.chars, stoi, w };
}
// ids(系列)を入れて、最後の位置のlogitsを返す
function lastLogits(m, ids) {
  const { n_embd: E, n_head: H, n_layer: L } = m.cfg, w = m.w, T = ids.length, D = E / H;
  let x = new Float32Array(T * E);
  for (let t = 0; t < T; t++) for (let i = 0; i < E; i++)
    x[t * E + i] = w['tok.weight'][ids[t] * E + i] + w['pos.weight'][t * E + i];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    const h = layerNorm(x, T, E, w[p + 'ln1.weight'], w[p + 'ln1.bias']);
    const qkv = linear(h, T, E, w[p + 'attn.in_proj_weight'], w[p + 'attn.in_proj_bias'], 3 * E);
    const att = new Float32Array(T * E);
    for (let hd = 0; hd < H; hd++) for (let t = 0; t < T; t++) {
      const sc = new Float32Array(t + 1); let mx = -Infinity;
      for (let s = 0; s <= t; s++) {
        let d = 0;
        for (let i = 0; i < D; i++) d += qkv[t * 3 * E + hd * D + i] * qkv[s * 3 * E + E + hd * D + i];
        sc[s] = d / Math.sqrt(D); if (sc[s] > mx) mx = sc[s];
      }
      let sum = 0; for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
      for (let s = 0; s <= t; s++) {
        const a = sc[s] / sum;
        for (let i = 0; i < D; i++) att[t * E + hd * D + i] += a * qkv[s * 3 * E + 2 * E + hd * D + i];
      }
    }
    const o = linear(att, T, E, w[p + 'attn.out_proj.weight'], w[p + 'attn.out_proj.bias'], E);
    for (let i = 0; i < x.length; i++) x[i] += o[i];
    const h2 = layerNorm(x, T, E, w[p + 'ln2.weight'], w[p + 'ln2.bias']);
    const f1 = linear(h2, T, E, w[p + 'ff.0.weight'], w[p + 'ff.0.bias'], 4 * E);
    for (let i = 0; i < f1.length; i++) f1[i] = gelu(f1[i]);
    const f2 = linear(f1, T, 4 * E, w[p + 'ff.2.weight'], w[p + 'ff.2.bias'], E);
    for (let i = 0; i < x.length; i++) x[i] += f2[i];
  }
  const last = x.slice((T - 1) * E, T * E);
  const hn = layerNorm(last, 1, E, w['ln.weight'], w['ln.bias']);
  return linear(hn, 1, E, w['head.weight'], w['head.bias'], m.chars.length);
}
function sample(logits, temperature, topK) {
  const n = logits.length, idx = [...Array(n).keys()].sort((a, b) => logits[b] - logits[a]).slice(0, Math.min(topK, n));
  const mx = logits[idx[0]]; const pr = idx.map(i => Math.exp((logits[i] - mx) / temperature));
  const sum = pr.reduce((a, b) => a + b, 0); let r = Math.random() * sum;
  for (let k = 0; k < idx.length; k++) { r -= pr[k]; if (r <= 0) return idx[k]; }
  return idx[0];
}
function reply(m, msg, opt = {}) {
  const { temperature = 0.7, topK = 20, maxNew = 120 } = opt, B = m.cfg.block_size;
  const ids = [...`ユーザー:${msg}\nAI:`].map(c => m.stoi[c] ?? 0);
  let out = '';
  for (let n = 0; n < maxNew; n++) {
    const ch = m.chars[sample(lastLogits(m, ids.slice(-B)), temperature, topK)];
    if (ch === '\n') break;
    out += ch; ids.push(m.stoi[ch]);
  }
  return out;
}


// ---- 教えたデータの管理 ----
const clean = (s) => String(s).replace(/[\r\n]+/g, ' ').trim();
function buildExport(items) {
  return items.filter(it => !it.sent).map(it => `ユーザー:${clean(it.q)}\nAI:${clean(it.a)}\n`).join('');
}
const b64enc = (str) => { let bin = ''; new TextEncoder().encode(str).forEach(x => bin += String.fromCharCode(x)); return btoa(bin); };
const b64dec = (b64) => new TextDecoder().decode(Uint8Array.from(atob(b64.replace(/\n/g, '')), c => c.charCodeAt(0)));

// 未送信の「教えた内容」を、GitHubの data.txt の末尾に1コミットで追記する（→ Actionsが自動で再学習）
async function appendToGitHub({ repo, token, items, fetchFn = fetch }) {
  const add = buildExport(items);
  if (!add) throw new Error('送る内容がありません');
  const url = `https://api.github.com/repos/${repo}/contents/data.txt`;
  const headers = { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' };
  const g = await fetchFn(url, { headers });
  if (!g.ok) throw new Error(`data.txt の取得に失敗 (${g.status})。リポジトリ名とトークンの権限を確認してください`);
  const cur = await g.json();
  let text = b64dec(cur.content); if (text && !text.endsWith('\n')) text += '\n';
  const p = await fetchFn(url, {
    method: 'PUT', headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: `teach: ${items.filter(i => !i.sent).length}件追加`, content: b64enc(text + add), sha: cur.sha })
  });
  if (!p.ok) throw new Error(`書き込みに失敗 (${p.status})`);
}

if (typeof module !== 'undefined') module.exports = { loadModel, lastLogits, reply, buildExport, appendToGitHub };
if (typeof document !== 'undefined') {
  const $ = (id) => document.getElementById(id);
  const log = $('log'), form = $('form'), input = $('input'), btn = $('send');
  const KEY = 'tiny-chat-ai-taught';
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; } };
  const save = (v) => { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) {} };
  let taught = load();
  const countEl = $('count');
  const updateCount = () => { countEl.textContent = taught.filter(t => !t.sent).length; };
  updateCount();

  const add = (who, text) => {
    const d = document.createElement('div'); d.className = 'msg ' + who; d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
  };
  const teach = (q, a) => {
    taught = taught.filter(t => clean(t.q) !== clean(q)); // 同じ質問は上書き
    taught.push({ q: clean(q), a: clean(a), sent: false }); save(taught); updateCount();
  };
  const addTeachButton = (bubble, q) => {
    const b = document.createElement('button'); b.className = 'teach'; b.textContent = '教える';
    b.onclick = () => {
      const a = prompt(`「${q}」への正しい返事を入力してください`);
      if (a && a.trim()) { teach(q, a); b.textContent = '覚えました ✓'; b.disabled = true; }
    };
    bubble.appendChild(document.createElement('br')); bubble.appendChild(b);
  };

  let model = null;
  const status = add('ai', 'モデルを読み込み中…');
  fetch('model.json').then(r => r.json()).then(j => {
    model = loadModel(j); status.textContent = '準備できました。話しかけてみてください！';
  }).catch(e => { status.textContent = '読み込みに失敗しました: ' + e; });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const msg = input.value.trim(); if (!msg || !model) return;
    input.value = ''; add('me', msg); btn.disabled = true;
    const thinking = add('ai', '…');
    setTimeout(() => {
      const hit = taught.find(t => t.q === clean(msg)); // 教えた質問には即答
      thinking.textContent = hit ? hit.a : (reply(model, msg) || '（……）');
      addTeachButton(thinking, msg);
      btn.disabled = false; input.focus(); log.scrollTop = log.scrollHeight;
    }, 30);
  });

  // 書き出し
  const dlg = $('export'), box = $('exportText');
  $('openExport').onclick = () => { box.value = buildExport(taught); dlg.hidden = false; };
  $('closeExport').onclick = () => { dlg.hidden = true; };
  $('copyExport').onclick = async () => {
    try { await navigator.clipboard.writeText(box.value); } catch (e) { box.select(); document.execCommand('copy'); }
    $('copyExport').textContent = 'コピーしました ✓'; setTimeout(() => $('copyExport').textContent = 'コピー', 1500);
  };
  $('pushGithub').onclick = async () => {
    const pending = taught.filter(t => !t.sent);
    if (!pending.length) { alert('送る内容がありません'); return; }
    let repo = localStorage.getItem('tca-repo'), token = localStorage.getItem('tca-token');
    if (!repo) { repo = prompt('GitHubのリポジトリ名（例: ユーザー名/tiny-chat-ai）', 'masa64xmasa-hub/tiny-chat-ai'); }
    if (!token) { token = prompt('GitHubのトークン（この端末のブラウザにだけ保存されます）'); }
    if (!repo || !token) return;
    const b = $('pushGithub'); b.disabled = true; b.textContent = '送信中…';
    try {
      await appendToGitHub({ repo: repo.trim(), token: token.trim(), items: taught });
      localStorage.setItem('tca-repo', repo.trim()); localStorage.setItem('tca-token', token.trim());
      taught.forEach(t => t.sent = true); save(taught); updateCount(); box.value = '';
      alert('GitHubに反映しました。数分で再学習され、公開ページが入れ替わります。');
    } catch (e) { alert('失敗しました: ' + e.message); }
    b.disabled = false; b.textContent = 'GitHubに反映';
  };
  $('resetToken').onclick = () => { localStorage.removeItem('tca-token'); localStorage.removeItem('tca-repo'); alert('保存したトークンを消しました'); };
  $('clearTaught').onclick = () => {
    if (confirm('教えた内容をすべて消しますか？（即答の記憶も消えます）')) {
      taught = []; save(taught); updateCount(); box.value = '';
    }
  };
}
