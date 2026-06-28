"""Render a replay document (from ``recorder.record_game``) to a standalone
HTML file. The game data is embedded directly in the page, so the result is a
single file you can open in any browser with no server and no dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hanabi Replay</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, Segoe UI, Roboto, system-ui, sans-serif;
    background: #11141b; color: #e7ebf2; padding: 16px 20px 40px;
  }
  h1 { font-size: 18px; margin: 0 0 2px; }
  .meta { color: #9aa6b8; font-size: 13px; margin-bottom: 14px; }
  .panel { background: #1a1f2b; border: 1px solid #283142; border-radius: 10px;
           padding: 12px 14px; margin-bottom: 14px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; }
  .stat { font-variant-numeric: tabular-nums; }
  .stat b { color: #fff; }
  .pip { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
         margin-right: 3px; border: 1px solid #44506a; vertical-align: middle; }
  .pip.on { background: #4dd08a; border-color: #4dd08a; }
  .pip.strike.on { background: #e23b3b; border-color: #e23b3b; }
  .controls button { background: #2a3344; color: #e7ebf2; border: 1px solid #3a465c;
    border-radius: 7px; padding: 6px 12px; font-size: 14px; cursor: pointer; }
  .controls button:hover { background: #344056; }
  .controls input[type=range] { flex: 1; min-width: 200px; }
  .hint { color: #6f7c91; font-size: 12px; }

  .stacks { display: flex; gap: 8px; }
  .card { width: 46px; height: 62px; border-radius: 8px; display: flex;
    flex-direction: column; align-items: center; justify-content: center;
    font-weight: 700; color: #fff; position: relative;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.18); }
  .card .rank { font-size: 22px; line-height: 1; }
  .card .know { font-size: 10px; font-weight: 600; margin-top: 4px;
    background: rgba(0,0,0,.32); border-radius: 4px; padding: 1px 4px; letter-spacing: .5px; }
  .card.empty { background: #232a38; box-shadow: inset 0 0 0 1px #2f3a4d; }
  .card.clued { box-shadow: 0 0 0 2px #f2c84b, inset 0 0 0 1px rgba(255,255,255,.2); }
  .card.touched { box-shadow: 0 0 0 3px #56b6ff, inset 0 0 0 1px rgba(255,255,255,.2); }
  .card.small { width: 30px; height: 40px; }
  .card.small .rank { font-size: 14px; }

  .hand { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
  .hand .label { width: 86px; color: #9aa6b8; font-size: 13px; }
  .hand.current .label { color: #f2c84b; font-weight: 700; }
  .hand.current { background: rgba(242,200,75,.06); border-radius: 8px; padding: 4px 6px; }
  .slot { display: flex; flex-direction: column; align-items: center; gap: 3px; }
  .slot .pos { font-size: 10px; color: #5f6c82; }

  .action { font-size: 15px; padding: 8px 12px; border-radius: 8px;
    background: #222b3a; border-left: 4px solid #56b6ff; }
  .action.play { border-left-color: #4dd08a; }
  .action.strike { border-left-color: #e23b3b; background: #2c2024; }
  .action.discard { border-left-color: #8a93a6; }
  .discard-wrap { display: flex; flex-wrap: wrap; gap: 5px; }
  .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: .6px;
    color: #7b8aa3; margin: 0 0 8px; }
</style>
</head>
<body>
  <h1>Hanabi Replay</h1>
  <div class="meta" id="meta"></div>

  <div class="panel">
    <div class="row controls" style="margin-bottom:10px">
      <button id="first">&#9198;</button>
      <button id="prev">&#9664; Prev</button>
      <button id="next">Next &#9654;</button>
      <button id="last">&#9197;</button>
      <input type="range" id="slider" min="0" value="0">
      <span class="stat" id="turnlabel"></span>
    </div>
    <div class="row">
      <span class="stat">Score <b id="score"></b>/<span id="maxscore"></span></span>
      <span class="stat">Clues <span id="tokens"></span></span>
      <span class="stat">Strikes <span id="strikes"></span></span>
      <span class="stat">Deck <b id="deck"></b></span>
    </div>
    <div class="hint">Use &larr;/&rarr; arrows or Home/End to step through.</div>
  </div>

  <div class="panel">
    <p class="section-title">Played stacks</p>
    <div class="stacks" id="stacks"></div>
  </div>

  <div class="panel">
    <div class="action" id="action"></div>
  </div>

  <div class="panel">
    <p class="section-title">Hands &nbsp;<span class="hint">(true card on top; small badge = what the holder knows)</span></p>
    <div id="hands"></div>
  </div>

  <div class="panel">
    <p class="section-title">Discard pile</p>
    <div class="discard-wrap" id="discard"></div>
  </div>

<script>
const DATA = __DATA__;
const CMAP = {R:'#e23b3b', Y:'#caa204', G:'#2ca34a', B:'#3b6fe2', P:'#9b4fd0'};
const CNAME = {R:'Red', Y:'Yellow', G:'Green', B:'Blue', P:'Purple'};
const COLORS = DATA.config.colors;
const FRAMES = DATA.frames;
let idx = 0;

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function knowBadge(card) {
  // card = {card:"R3", pc:[...], pr:[...], clued}
  const kc = card.pc.length === 1 ? card.pc[0] : '?';
  const kr = card.pr.length === 1 ? String(card.pr[0]) : '?';
  return kc + kr;
}

function cardTile(card, opts) {
  opts = opts || {};
  const color = card.card[0], rank = card.card[1];
  const d = el('div', 'card' + (opts.small ? ' small' : ''));
  d.style.background = CMAP[color] || '#444';
  if (card.clued) d.classList.add('clued');
  if (opts.touched) d.classList.add('touched');
  d.appendChild(el('span', 'rank', rank));
  if (!opts.small) d.appendChild(el('span', 'know', knowBadge(card)));
  d.title = 'True: ' + (CNAME[color] || color) + ' ' + rank +
    '\nKnows colors: ' + card.pc.join(',') + '\nKnows ranks: ' + card.pr.join(',') +
    (card.clued ? '\n(clued)' : '\n(unclued)');
  return d;
}

function pips(n, max, cls) {
  let s = '';
  for (let i = 0; i < max; i++) s += '<span class="pip ' + (cls||'') + (i < n ? ' on' : '') + '"></span>';
  return s;
}

function render() {
  const f = FRAMES[idx];
  const action = f.action;
  const touched = new Set(action ? action.touched : []);

  document.getElementById('turnlabel').textContent = 'Frame ' + idx + ' / ' + (FRAMES.length - 1);
  document.getElementById('score').textContent = f.score;
  document.getElementById('deck').textContent = f.deck_size;
  document.getElementById('tokens').innerHTML = pips(f.clue_tokens, DATA.config.max_clue_tokens) +
    ' <b>' + f.clue_tokens + '</b>';
  document.getElementById('strikes').innerHTML = pips(f.strikes, DATA.config.max_strikes, 'strike') +
    ' <b>' + f.strikes + '</b>';

  // Stacks
  const stacks = document.getElementById('stacks'); stacks.innerHTML = '';
  COLORS.forEach(c => {
    const top = f.stacks[c] || 0;
    if (top === 0) {
      const e = el('div', 'card empty');
      e.appendChild(el('span', 'rank', c));
      stacks.appendChild(e);
    } else {
      stacks.appendChild(cardTile({card: c + top, pc:[c], pr:[top], clued:false}));
    }
  });

  // Action banner
  const a = document.getElementById('action');
  if (!action) {
    a.className = 'action'; a.textContent = 'Initial deal';
  } else {
    let cls = 'action';
    if (action.type === 'play') cls += action.success ? ' play' : ' strike';
    else if (action.type === 'discard') cls += ' discard';
    a.className = cls; a.textContent = action.text;
  }

  // Hands
  const hands = document.getElementById('hands'); hands.innerHTML = '';
  f.hands.forEach((hand, p) => {
    const row = el('div', 'hand' + (p === f.current_player && !f.game_over ? ' current' : ''));
    row.appendChild(el('div', 'label', 'P' + p + (p === f.current_player && !f.game_over ? ' ▶' : '')));
    // Display newest-first (left = newest = slot 1), matching the draw-to-the-
    // left convention. Order IDs are unchanged; this is display only.
    const ordered = hand.slice().reverse();
    ordered.forEach((card, i) => {
      const wrap = el('div', 'slot');
      wrap.appendChild(cardTile(card, {touched: touched.has(card.order)}));
      const tag = i === 0 ? 'new' : (i === ordered.length - 1 ? 'old' : '');
      wrap.appendChild(el('div', 'pos', tag));
      row.appendChild(wrap);
    });
    hands.appendChild(row);
  });

  // Discard (sorted by color then rank)
  const disc = document.getElementById('discard'); disc.innerHTML = '';
  const sorted = f.discard.slice().sort((x, y) =>
    COLORS.indexOf(x[0]) - COLORS.indexOf(y[0]) || x[1].localeCompare(y[1]));
  if (sorted.length === 0) disc.appendChild(el('span', 'hint', 'empty'));
  sorted.forEach(cs => disc.appendChild(cardTile({card: cs, pc:[cs[0]], pr:[Number(cs[1])], clued:false}, {small:true})));

  document.getElementById('slider').value = idx;
}

function go(i) { idx = Math.max(0, Math.min(FRAMES.length - 1, i)); render(); }

document.getElementById('first').onclick = () => go(0);
document.getElementById('prev').onclick = () => go(idx - 1);
document.getElementById('next').onclick = () => go(idx + 1);
document.getElementById('last').onclick = () => go(FRAMES.length - 1);
document.getElementById('slider').oninput = (e) => go(Number(e.target.value));
document.addEventListener('keydown', (e) => {
  const targets = {ArrowRight: idx + 1, ArrowLeft: idx - 1, Home: 0, End: FRAMES.length - 1};
  if (!(e.key in targets)) return;
  // Stop the focused slider from ALSO moving (which would advance two frames).
  e.preventDefault();
  go(targets[e.key]);
});

(function init() {
  const r = DATA.result;
  const outcome = r.strikeout ? 'STRIKEOUT (scored 0)' : (r.won ? 'PERFECT 25!' : 'final score ' + r.stack_total);
  document.getElementById('meta').textContent =
    DATA.config.num_players + ' players · strategy: ' + DATA.strategy +
    ' · seed ' + DATA.seed + ' · ' + outcome +
    ' · ' + (FRAMES.length - 1) + ' actions';
  document.getElementById('maxscore').textContent = DATA.config.max_score;
  document.getElementById('slider').max = FRAMES.length - 1;
  render();
})();
</script>
</body>
</html>
"""


def render_html(replay: dict) -> str:
    """Return a standalone HTML document for the given replay."""
    data_json = json.dumps(replay, separators=(",", ":"))
    return _TEMPLATE.replace("__DATA__", data_json)


def write_replay(path: str | Path, replay: dict) -> Path:
    out = Path(path)
    out.write_text(render_html(replay), encoding="utf-8")
    return out
