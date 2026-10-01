import {Game2048, Connect4, chooseConnect4, winningCells, legalColumns} from './game-engines.mjs';

const $ = id => document.getElementById(id);
const games = {
  tetris: {name: 'Tetris', title: 'A placement game, with a clear target.', copy: 'Models choose a complete legal landing. The supplied policy prioritizes fewer holes, then cleared rows, peak height, total height and surface roughness.', empty: () => Array.from({length: 20}, () => Array(10).fill(0))},
  '2048': {name: '2048', title: 'A simple move can close off the next one.', copy: 'The measured rule maximizes immediate merge score, then empty cells, then the largest tile, before a random tile spawns. It is a one-step policy, not an optimal full-game strategy.', empty: () => Array.from({length: 4}, () => Array(4).fill(0))},
  connect4: {name: 'Connect Four', title: 'Plan your line. Notice theirs.', copy: 'Measured models choose the column with the highest minimax score, searching four further plies after their proposed move. Capped recorded games use a deterministic four-ply opponent.', empty: () => Array.from({length: 6}, () => Array(7).fill(0))},
  blockstar: {name: 'BlockStar', title: 'Move the pieces into their target pattern.', copy: 'An adaptation of the original BlockStar puzzle. Slide labeled blocks into the target layout, respecting the puzzle rules and ordered-piece policy. These compact benchmark fixtures are documented with the source.', empty: () => []},
};
const GAME_ORDER = ['tetris','2048','connect4','blockstar'];
const state = {game: 'tetris', mode: 'replay', models: [], replayModels: [], selectedModel: null, selectedRun: 0, run: null, index: 0, timer: null, human: null, humanGeneration: 0, source: null, summary: null, tetrisSummary: null, updateSummary: null, updateSource: null};
const pieceColors = {I:'#81bfc3',O:'#d3bf77',T:'#ada0cb',S:'#a1bd7e',Z:'#c68f84',J:'#84a1c7',L:'#cbaa7e'};
const blockColors = ['#81a38d','#b4bd7c','#7eb5b0','#b5a0c1','#c2a281','#87a2c3','#bd8686','#a1b885'];
const number = value => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value) : null;
const pct = value => number(value) === null ? '—' : `${Number(value).toFixed(1).replace(/\.0$/, '')}%`;
const ms = value => number(value) === null ? '—' : Number(value) >= 1000 ? `${(Number(value) / 1000).toFixed(2)} s` : `${Math.round(Number(value))} ms`;
const humanize = value => String(value ?? '').replace(/_/g, ' ');
function element(tag, className, content) {const node = document.createElement(tag); if (className) node.className = className; if (content !== undefined) node.textContent = content; return node;}
function boardRows(board) {return Array.isArray(board) ? board.map(row => Array.isArray(row) ? row : typeof row === 'string' ? [...row] : []) : [];}
function metric(model, game = state.game) {const data = model?.games?.[game]; if (!data) return null; return {...data, median_ms: data.fresh_timing?.median_ms ?? data.median_ms ?? data.median_placement_ms, p95_ms: data.fresh_timing?.p95_ms ?? data.p95_ms ?? data.p95_placement_ms};}
function rowsSorted() {return state.models.slice().sort((a,b) => (number(metric(b)?.agreement_pct) ?? -1) - (number(metric(a)?.agreement_pct) ?? -1) || (number(metric(a)?.median_ms) ?? Infinity) - (number(metric(b)?.median_ms) ?? Infinity));}
function stopPlayback() {clearInterval(state.timer); state.timer = null; $('play-button').replaceChildren(element('span', '', '▶'), document.createTextNode(' Play replay'));}

async function getJSON(path) {const response = await fetch(path, {cache:'no-store'}); if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`); return response.json();}
async function loadData() {
  const paths = ['data/tetris-summary.json','data/tetris-replays.json','data/arcade-summary.json','data/arcade-replays.json','data/blockstar-source.json','data/blockstar-update-summary.json','data/blockstar-update-source.json'];
  const responses = await Promise.allSettled(paths.map(getJSON));
  const data = responses.map(result => result.status === 'fulfilled' ? result.value : null);
  const [tetris, tetrisReplay, arcade, arcadeReplay, source, updateSummary, updateSource] = data;
  state.summary = arcade; state.tetrisSummary = tetris; state.source = source;
  state.updateSummary = updateSummary;state.updateSource = updateSource;
  const merged = new Map();
  for (const model of tetris?.models || []) {
    merged.set(model.name, {...model, games:{tetris: model.summary || model.games?.tetris}, origin: model.origin || 'tetris'});
  }
  for (const model of arcade?.models || []) {
    const previous = merged.get(model.name) || {name:model.name,label:model.label,games:{}};
    merged.set(model.name, {...previous,...model,games:{...previous.games,...model.games}});
  }
  state.models = [...merged.values()];
  const replayMerged = new Map();
  for (const [bundle, defaultGame] of [[tetrisReplay,'tetris'],[arcadeReplay,null]]) for (const model of bundle?.models || []) {
    const previous = replayMerged.get(model.name) || {name:model.name,label:model.label,games:[]};
    previous.games.push(...(model.games || []).map(game => ({...game,game:game.game || defaultGame})));
    replayMerged.set(model.name, previous);
  }
  state.replayModels = [...replayMerged.values()];
  $('model-count').textContent = state.models.length || '—';
  const decisions = state.models.reduce((sum,model) => sum + Object.values(model.games).reduce((count,summary) => count + (number(summary?.valid) ?? 0),0),0) + (updateSummary?.models || []).reduce((sum,model) => sum + Object.values(model.summary || {}).reduce((count,summary) => count + (number(summary?.valid) ?? 0),0),0);
  $('decision-count').textContent = decisions ? decisions.toLocaleString() : '—';
  if (arcade?.hardware || tetris?.hardware) $('hardware').textContent = arcade?.hardware || tetris.hardware;
  const missing = responses.slice(0,5).filter(response => response.status === 'rejected').length;
  const arcadeStatus = arcade?.status ? `New games: ${humanize(arcade.status)}.` : 'New game inference has not been published.';
  $('data-status').textContent = state.models.length ? `${state.models.length} runtimes · ${decisions.toLocaleString()} valid test decisions. ${arcadeStatus}${missing ? ' Some evidence files are unavailable.' : ''}` : 'Published evidence is not available yet. The browser games are ready to play.';
  $('data-status').classList.toggle('error',missing === 5);
  if (source) {
    $('blockstar-source-copy').textContent = source.description || source.adaptation?.description || source.method || 'BlockStar rules are adapted from the original source into explicit legal-slide decisions. The target layout is shown with each replay; source identity and changes are recorded below.';
    $('blockstar-source-detail').textContent = JSON.stringify(source,null,2);
  } else $('blockstar-source-detail').textContent = 'The source snapshot has not been published yet. Visit the original repository for its rules and implementation.';
  renderLeaderboard(); renderMethod(); populateModels();
}

function selectGame(game) {
  stopPlayback(); state.game = game; state.index = 0; state.humanGeneration++;
  if (!['2048','connect4'].includes(game)) state.mode = 'replay';
  document.querySelectorAll('.game-tab').forEach(tab => {const active = tab.dataset.game === game; tab.classList.toggle('active',active); tab.setAttribute('aria-selected',String(active)); tab.tabIndex = active ? 0 : -1;});
  $('game-panel').setAttribute('aria-labelledby',`tab-${game}`);
  $('replay-selectors').classList.toggle('blockstar-selectors',game === 'blockstar');
  $('mode-human').hidden = !['2048','connect4'].includes(game);
  $('context-title').textContent = games[game].title; $('context-copy').textContent = games[game].copy;
  setMode(state.mode); renderLeaderboard(); renderMethod();
}
function setMode(mode) {
  stopPlayback(); state.mode = mode; state.humanGeneration++;
  const human = mode === 'human';
  $('mode-replay').classList.toggle('selected',!human); $('mode-replay').setAttribute('aria-pressed',String(!human));
  $('mode-human').classList.toggle('selected',human); $('mode-human').setAttribute('aria-pressed',String(human));
  for (const id of ['replay-controls','replay-selectors','board-progress','inference-metrics','move-details']) $(id).hidden = human;
  for (const id of ['human-controls','human-description']) $(id).hidden = !human;
  $('direction-pad').hidden = !human || state.game !== '2048'; $('c4-columns').hidden = !human || state.game !== 'connect4';
  $('workspace-tag').replaceChildren(element('span','','●'),document.createTextNode(human ? ' BROWSER GAME ENGINE' : ' SAVED INFERENCE'));
  $('board-mode').textContent = human ? 'HUMAN PLAY' : 'REPLAY';
  $('decision-eyebrow').textContent = human ? 'YOUR GAME' : 'THE LAST DECISION';
  $('board').setAttribute('role',human ? 'application' : 'img');
  if (human) startHuman(); else {$('context-title').textContent = games[state.game].title;$('context-copy').textContent = games[state.game].copy;populateModels();}
}
function populateModels() {
  const playable = state.replayModels.filter(model => model.games.some(game => game.game === state.game && Array.isArray(game.moves) && game.moves.length));
  const previous = state.selectedModel;
  $('model-select').replaceChildren();
  playable.sort((a,b) => (number(metric(state.models.find(model => model.name === b.name))?.agreement_pct) ?? -1) - (number(metric(state.models.find(model => model.name === a.name))?.agreement_pct) ?? -1));
  for (const model of playable) {const option = element('option','',model.label || model.name); option.value = model.name; $('model-select').append(option);}
  if (!playable.length) {const option = element('option','','No saved model runs yet'); option.value = ''; $('model-select').append(option); state.selectedModel = null;}
  else {state.selectedModel = playable.some(model => model.name === previous) ? previous : playable[0].name; $('model-select').value = state.selectedModel;}
  $('model-select').disabled = !playable.length; populateRuns();
}
function populateRuns() {
  stopPlayback(); state.index = 0; state.selectedRun = 0;
  const runs = state.replayModels.find(model => model.name === state.selectedModel)?.games.filter(game => game.game === state.game) || [];
  $('run-select').replaceChildren();
  runs.forEach((run,index) => {const option = element('option','',run.label || `Seed ${run.seed ?? index + 1}`); option.value = String(index); $('run-select').append(option);});
  if (!runs.length) $('run-select').append(element('option','','No runs'));
  $('run-select').disabled = !runs.length; state.run = runs[0] || null;
  renderReplay();
}
function chooseRun() {stopPlayback(); state.index = 0; state.selectedRun = Number($('run-select').value); state.run = state.replayModels.find(model => model.name === state.selectedModel)?.games.filter(game => game.game === state.game)[state.selectedRun] || null; renderReplay();}
function nextMove() {if (!state.run || state.index >= state.run.moves.length) {stopPlayback(); return;} state.index++; renderReplay(); if (state.index >= state.run.moves.length) stopPlayback();}
function togglePlayback() {if (state.timer) {stopPlayback();return;} if (!state.run?.moves?.length) return; if (state.index >= state.run.moves.length) state.index = 0; $('play-button').replaceChildren(element('span','','Ⅱ'),document.createTextNode(' Pause replay')); state.timer = setInterval(nextMove,650); nextMove();}

function drawBoard(snapshot, human = false) {
  const board = $('board'); const rows = boardRows(snapshot?.board); board.replaceChildren();
  board.className = `game-board ${state.game === 'tetris' ? 'tetris-board' : state.game === '2048' ? 'tile-board' : state.game === 'connect4' ? 'c4-board' : 'blockstar-layout'}`;
  if (!rows.length) {board.className = 'pending-board'; board.textContent = 'No saved board for this game yet.\nMeasured frames will appear here after inference is published.'; board.setAttribute('aria-label',board.textContent);return;}
  if (state.game === 'blockstar') {
    const labels = [...new Set(rows.flat().filter(value => value && value !== '-' && value !== '.' && value !== '#'))].sort();
    function blockGrid(grid,label,small) {
      const wrapper = element('div',small ? 'target-board' : 'current-board'); wrapper.append(element('p','mini-label',label));
      const cells = element('div','blockstar-grid'); cells.style.setProperty('--cols',String(grid[0]?.length || 1));
      cells.classList.toggle('dense-board',(grid[0]?.length || 0) > 20);
      for (const row of grid) for (const value of row) {const cell = element('div','board-cell',value === '-' || value === '.' || value === '#' ? '' : value); cell.setAttribute('aria-hidden','true'); if (value === '#') {cell.classList.add('fixed-terrain');cell.title = 'Permanent terrain: cannot move or be crossed';} else if (value && value !== '-' && value !== '.') {const index = labels.indexOf(value); cell.style.background = blockColors[(Math.max(0,index)) % blockColors.length]; cell.style.color = '#102014';} cells.append(cell);}
      wrapper.append(cells); return wrapper;
    }
    board.append(blockGrid(rows,'CURRENT LAYOUT',false));
    const target = boardRows(snapshot.target || state.run?.initial?.target);
    if (target.length) board.append(blockGrid(target,'TARGET PATTERN',true));
    board.setAttribute('aria-label',`BlockStar current board: ${rows.map(row => row.join(' ')).join('; ')}. '#' cells are permanent terrain; they cannot move or be crossed. ${target.length ? `Target: ${target.map(row => row.join(' ')).join('; ')}.` : ''}`);return;
  }
  board.style.setProperty('--cols',String(rows[0]?.length || 4));
  const winning = state.game === 'connect4' ? winningCells(rows)?.cells || [] : [];
  rows.forEach((row,y) => row.forEach((value,x) => {
    const cell = element('div','board-cell'); cell.setAttribute('aria-hidden','true');
    if (state.game === 'tetris' && value && value !== '.' && value !== '-') {cell.style.background = pieceColors[value] || '#a1bd7e'; cell.style.boxShadow = 'inset 0 0 0 1px #ffffff18';}
    if (state.game === '2048') {if (value) {cell.textContent = value; const exponent = Math.log2(Number(value)); cell.style.background = ['#344638','#d5ddc1','#becda4','#a9c487','#8aae66','#719e51','#ccbc6e','#d3a95d','#ce8e50','#c17649','#b76343','#b35b44'][Math.min(11,exponent)] || '#b35b44'; cell.style.color = exponent <= 4 ? '#1a2b1b' : '#fff3da';if (value >= 1024) cell.classList.add('tile-large');} else cell.classList.add('empty');}
    if (state.game === 'connect4') {if (value) cell.classList.add(Number(value) === 1 ? 'player-one' : 'player-two'); if (winning.some(([wy,wx]) => wy === y && wx === x)) cell.classList.add('winning');}
    board.append(cell);
  }));
  const label = state.game === '2048' ? `2048 board, rows: ${rows.map(row => row.join(', ')).join('; ')}. Use arrow keys or WASD to move.` : state.game === 'connect4' ? `Connect Four board: ${rows.map(row => row.map(value => value === 1 ? 'you' : value === 2 ? 'opponent' : 'empty').join(', ')).join('; ')}. ${human ? 'Choose a column using the buttons or number keys 1 to 7.' : ''}` : `Tetris board with ${rows.flat().filter(Boolean).length} occupied cells.`;
  board.setAttribute('aria-label',label);
}
function setStats(first,label1,second,label2,third,label3) {for (const [key,value,label] of [['one',first,label1],['two',second,label2],['three',third,label3]]) {$(`stat-${key}-value`).textContent = value ?? '—'; $(`stat-${key}-label`).textContent = label;}}
function snapshotStats(snapshot) {
  if (state.game === 'tetris') setStats(snapshot.pieces ?? state.index,'PIECES',snapshot.lines ?? 0,'LINES',snapshot.score ?? 0,'SCORE');
  if (state.game === '2048') setStats(snapshot.score ?? 0,'SCORE',snapshot.max_tile ?? (Math.max(0,...boardRows(snapshot.board).flat().map(Number)) || 0),'MAX TILE',snapshot.turn ?? state.index,'MOVES');
  if (state.game === 'connect4') setStats(snapshot.turn ?? state.index,state.mode === 'replay' ? 'MODEL TURNS' : 'MOVES',snapshot.winner ? humanize(snapshot.winner) : snapshot.player ?? snapshot.to_move ?? '—','PLAYER',snapshot.status ? humanize(snapshot.status) : 'playing','STATUS');
  if (state.game === 'blockstar') {const rows = boardRows(snapshot.board),target = boardRows(snapshot.target || state.run?.initial?.target);const correct = rows.reduce((sum,row,y) => sum + row.filter((value,x) => value && value !== '-' && value !== '.' && value !== '#' && value === target[y]?.[x]).length,0);setStats(snapshot.turn ?? state.index,'MOVES',snapshot.distance_moved ?? '—','SLIDE DISTANCE',target.length ? correct : '—','CELLS ON TARGET');}
}
function renderReplay() {
  if (state.mode !== 'replay') return;
  const updatedReference = state.run?.scope === 'updated-official-source-baseline' || ['official-champion-certificate','updated-official-certificate','updated-source-solver','official-source-solver'].includes(state.run?.source);
  const reference = updatedReference || state.run?.scope === 'full-original-source-baseline' || state.run?.source === 'original-stochastic-solver';
  const oneStep = state.run?.scope === 'updated-official-one-step-probe';
  const move = state.index ? state.run?.moves[state.index - 1] : null;
  const snapshot = move?.after || move?.action?.after || state.run?.initial || {board:games[state.game].empty()};
  drawBoard(snapshot); snapshotStats(snapshot);
  const rows = boardRows(snapshot.board);
  const fullOriginal = state.game === 'blockstar' && rows.length === 20 && rows[0]?.length === 18;
  const updatedName = (snapshot.source_case || state.run?.scenario || '').includes('large-yard') ? 'LARGE YARD' : 'UPDATED DEFAULT';
  $('board-title').textContent = `${games[state.game].name.toUpperCase()} / ${state.run ? state.game === 'blockstar' ? `${rows.length} × ${rows[0]?.length || 0} ${oneStep || updatedReference ? `${updatedName} / ${oneStep ? 'FIXED PROBE' : 'REFERENCE'}` : `${fullOriginal ? 'ORIGINAL BOARD' : 'FIXTURE'} / SEED ${state.run.seed ?? '—'}`}` : `SEED ${state.run.seed ?? '—'}` : 'NO SAVED RUN'}`;
  $('model-selector-label').textContent = state.game === 'blockstar' ? 'MODEL / REFERENCE' : 'MODEL RUNTIME';
  $('model-select').setAttribute('aria-label',state.game === 'blockstar' ? 'Replay model or solver reference' : 'Replay model');
  $('workspace-tag').replaceChildren(element('span','','●'),document.createTextNode(updatedReference ? ' UPDATED SOURCE REFERENCE' : reference ? ' ORIGINAL SOLVER TRACE' : oneStep ? ' UPDATED ONE-STEP INFERENCE' : ' SAVED INFERENCE'));
  $('board-mode').textContent = reference ? 'REFERENCE' : oneStep ? 'ONE-STEP PROBE' : 'REPLAY';
  $('decision-eyebrow').textContent = updatedReference ? 'THE UPDATED REFERENCE SLIDE' : reference ? 'THE ORIGINAL SOLVER SLIDE' : oneStep ? 'THE FIXED-BOARD DECISION' : 'THE LAST DECISION';
  $('context-title').textContent = updatedReference ? 'An updated official reference, replay-validated.' : reference ? 'The original solver, on the full original board.' : oneStep ? 'One fixed board. One measured decision.' : games[state.game].title;
  $('context-copy').textContent = updatedReference ? 'This full-scenario reference records the updated source. Certificate replay verification and solution discovery are separate measurements. Its full-puzzle outcome is separate from model one-step probes; playback cadence is unrelated to either recorded runtime.' : reference ? 'This independently validated reference trace records the upstream stochastic solver. Its full-puzzle outcome is separate from measured model decisions and the compact model replay fixtures. Playback cadence is unrelated to solver wall time.' : oneStep ? 'This is a supplemental probe of an updated official initial scenario, with every legal slide passed to the model. The supplied sorted-piece prefix policy is an evaluation rule, not a legality requirement of the upstream solver. One step does not establish a full-puzzle solve rate.' : games[state.game].copy;
  const length = state.run?.moves?.length || 0;
  $('move-counter').textContent = `${state.index} / ${length}`;
  $('progress-fill').style.width = `${length ? state.index / length * 100 : 0}%`;
  for (const id of ['play-button','step-button','restart-button']) $(id).disabled = !length;
  $('step-button').disabled = !length || state.index >= length;
  $('board-instructions').textContent = updatedReference ? 'Verified updated-source reference. No model inference. Hatched # cells are fixed terrain.' : reference ? 'Recorded original solver on the full 20 × 18 board. No model inference.' : oneStep ? 'Updated official initial board / one-step probe. Hatched # cells cannot move or be crossed.' : state.game === 'blockstar' ? fullOriginal ? 'Recorded original 20 × 18 puzzle. Saved model inference only.' : 'Recorded compact fixture. It is separate from the full original 20 × 18 puzzle.' : 'Playback uses saved boards. It does not run a model in your browser.';
  $('inference-time').textContent = reference ? '—' : ms(move?.decision?.latency_ms); $('candidate-count').textContent = reference ? '—' : move?.decision?.candidate_count ?? '—';
  $('call-count').textContent = reference ? '—' : Array.isArray(move?.decision?.calls) ? move.decision.calls.length : move?.decision?.calls ?? '—';
  if (move) {
    const action = move.action || move.move || {},metrics = action.metrics || {};
    if (state.game === 'tetris') {$('decision-title').textContent = `${action.piece ?? 'Piece'} · column ${Number(action.x ?? 0)+1} · ${action.rotation ?? 0}°`; $('decision-detail').textContent = `${action.cleared ?? 0} rows cleared · ${metrics.holes ?? '—'} holes · peak height ${metrics.max_height ?? '—'}`;}
    else {$('decision-title').textContent = action.text || humanize(move.decision?.choice || action.id || snapshot.last_move || `Move ${state.index}`); const entries = Object.entries(metrics).filter(([,value]) => ['number','string','boolean'].includes(typeof value)).slice(0,5); $('decision-detail').textContent = entries.length ? entries.map(([key,value]) => `${humanize(key)}: ${value}`).join(' · ') : 'This action and resulting board come directly from the saved model trace.';}
    $('move-json').textContent = JSON.stringify(move,null,2);
  } else {
    $('decision-title').textContent = state.run ? 'Ready when you are.' : 'Awaiting a recorded run.';
    $('decision-detail').textContent = state.run ? updatedReference ? 'Replay this verified updated-source solution, slide by slide. Certificate verification time and any captured discovery runtime are reported separately below; per-slide model timing does not apply.' : reference ? 'Replay the original stochastic solver, slide by slide, on the full source puzzle. Total solver wall time is reported below; per-slide inference timing does not apply.' : oneStep ? 'Inspect one saved native-model decision on an official initial board. This probe preserves the complete legal action set and does not claim that the model solved the puzzle.' : `Replay ${state.replayModels.find(model => model.name === state.selectedModel)?.label || state.selectedModel}. Step through each saved decision, including its measured latency.` : 'Model inference for this game has not been published yet. Missing results are never replaced by a browser heuristic.';
    $('move-json').textContent = 'No move selected.';
  }
  const discovery = updatedReference ? state.run.discovery_elapsed_s ?? state.run.discovery_elapsed_seconds : reference ? state.run.elapsed_s : null;
  const verification = updatedReference ? state.run.replay_verification_s ?? state.run.verification_elapsed_s ?? state.run.certificate_replay_elapsed_seconds : null;
  const elapsed = `${number(discovery) !== null ? ` · total ${updatedReference ? 'discovery' : 'solver'} wall time: ${Number(discovery).toFixed(2)} s` : ''}${number(verification) !== null ? ` · replay verification: ${Number(verification).toFixed(4)} s` : ''}`;
  $('game-status').textContent = state.run ? `${state.index}/${length} saved ${reference ? 'slides' : 'moves'} · recorded outcome: ${humanize(state.run.status || 'unspecified')}${elapsed}` : 'No inference result available.';
}

function renderLeaderboard() {
  document.querySelectorAll('th[data-column]').forEach(node => node.classList.toggle('selected-column',node.dataset.column === state.game));
  const body = $('leaderboard-body'); body.replaceChildren();
  if (!state.models.length) {const row = element('tr'); const cell = element('td','table-empty','No measured model results have been published yet.');cell.colSpan = 5;row.append(cell);body.append(row);renderCharts();renderOriginalBlockstar();renderUpdatedBlockstar();return;}
  rowsSorted().forEach((model,index) => {
    const row = element('tr'); const name = element('td'); const wrapper = element('div','model-name'); wrapper.append(element('span','rank',String(index + 1).padStart(2,'0')));const label = element('div'); label.append(element('b','',model.label || model.name));
    const runtime = model.metadata?.models?.[0]?.backend || model.runtime || model.name;
    label.append(element('small','',runtime));wrapper.append(label);name.append(wrapper);row.append(name);
    for (const game of GAME_ORDER) {
      const cell = element('td',game === state.game ? 'selected-column' : ''); const summary = metric(model,game);
      if (number(summary?.agreement_pct) === null) {cell.append(element('span','no-result',summary?.status ? humanize(summary.status) : 'Not measured'),element('small','metric-meta',summary?.reason || '—'));}
      else {cell.append(element('span','agreement',pct(summary.agreement_pct)),element('small','metric-meta',`${ms(summary.median_ms)} median`)); const bar = element('div','agreement-bar'); const fill = element('span');fill.style.width = `${Math.min(100,Math.max(0,Number(summary.agreement_pct)))}%`;bar.append(fill);cell.append(bar);cell.title = `${summary.valid ?? '—'}/${summary.trials ?? '—'} valid trials; P95 ${ms(summary.p95_ms)}`;cell.setAttribute('aria-label',`${games[game].name}: ${pct(summary.agreement_pct)} agreement, ${ms(summary.median_ms)} median, ${cell.title}`);}
      row.append(cell);
    }
    body.append(row);
  });
  $('table-note').textContent = `Sorted by ${games[state.game].name} policy agreement. Each cell shows agreement and median decision latency; valid/trial counts and P95 are in the cell tooltip. New-game medians use first option-order passes, excluding exact warmup states; repeated inputs are recorded separately. Missing results remain unscored.`;
  renderCharts();renderOriginalBlockstar();renderUpdatedBlockstar();
}
function renderUpdatedBlockstar() {
  $('blockstar-updated').hidden = state.game !== 'blockstar';
  const data = state.updateSummary,source = state.updateSource;
  $('updated-status').textContent = data?.status ? `Supplement: ${humanize(data.status)}` : 'Evidence pending';
  const body = $('updated-model-body');body.replaceChildren();
  const models = data?.models || [];
  for (const model of models) {
    const row = element('tr');row.append(element('td','',model.label || model.name));
    for (const scenario of ['updated-default','large-yard']) {
      const cell = element('td'),summary = model.summary?.[scenario];
      if (number(summary?.agreement_pct) === null) cell.append(element('span','no-result','Pending inference'));
      else {cell.append(element('span','agreement',pct(summary.agreement_pct)),element('small','metric-meta',`${ms(summary.median_ms)} first order`),element('small','metric-meta',`${ms(summary.repeated_median_ms)} repeated order`),element('small','metric-meta',`${summary.valid ?? '—'}/${summary.trials ?? '—'} valid · ${summary.candidate_count ?? '—'} legal slides`));cell.setAttribute('aria-label',`${scenario}: ${pct(summary.agreement_pct)} agreement, ${ms(summary.median_ms)} first-order timing, ${ms(summary.repeated_median_ms)} repeated-order timing, ${summary.valid ?? '—'}/${summary.trials ?? '—'} valid`);}
      row.append(cell);
    }
    body.append(row);
  }
  if (!models.length) {const row = element('tr'),cell = element('td','table-empty','Updated scenario inference has not been published yet.');cell.colSpan = 3;row.append(cell);body.append(row);}
  $('updated-method').textContent = data?.method || 'Supplemental evidence is awaiting publication. Missing updated results leave the earlier measurements available.';
  const commit = data?.source_commit || source?.commit;
  $('updated-source-copy').replaceChildren();
  if (commit) {
    $('updated-source-copy').append(document.createTextNode('Updated source pin: '));
    const link = element('a','',commit.slice(0,12));if (/^[a-f0-9]{40}$/.test(commit)) link.href = `https://github.com/rayking99/BlockStar/tree/${commit}`;$('updated-source-copy').append(link,document.createTextNode('. The earlier experiment stays pinned to its original source. Fixed # terrain does not count toward movable-piece target progress.'));
  }
  const referenceResults = (source?.replay_reference || []).map(run => `${run.case_id || run.label || 'Official scenario'}: ${run.slide_count ?? '—'} validated slides${number(run.discovery_elapsed_seconds) !== null ? ` · ${Number(run.discovery_elapsed_seconds).toFixed(2)} s discovery` : ''}${number(run.certificate_replay_elapsed_seconds) !== null ? ` · ${Number(run.certificate_replay_elapsed_seconds).toFixed(4)} s certificate replay` : ''}`).join('; ');
  $('updated-reference-copy').textContent = `${referenceResults ? `${referenceResults}. ` : ''}Updated source solution certificates are references, not model results. Their slide counts are validated replay scores and are not asserted to be globally optimal. Certificate verification time is reported separately from measured solution-discovery runtime.`;
  $('updated-source-detail').textContent = source ? JSON.stringify(source,null,2) : 'No updated source record published yet.';
}
function renderOriginalBlockstar() {
  $('blockstar-original').hidden = state.game !== 'blockstar';
  const body = $('blockstar-original-body');body.replaceChildren();
  const measured = rowsSorted().filter(model => number(metric(model,'blockstar')?.original_agreement_pct) !== null);
  for (const model of measured) {const summary = metric(model,'blockstar'); const row = element('tr');row.append(element('td','',model.label || model.name),element('td','agreement',pct(summary.original_agreement_pct)),element('td','',ms(summary.original_median_ms)),element('td','',`${summary.original_valid ?? summary.original_trials ?? '—'}/${summary.original_trials ?? '—'}`));body.append(row);}
  if (!measured.length) {const row = element('tr');const cell = element('td','table-empty','Full original-board model inference has not been published yet.');cell.colSpan = 4;row.append(cell);body.append(row);}
  const baseline = state.source?.upstream_measured_baseline,summary = baseline?.summary;
  $('original-solver-note').replaceChildren();
  if (summary) {
    const solved = summary.solved ?? summary.solved_runs ?? baseline.trials?.filter(trial => trial.solved).length;
    const total = summary.runs ?? summary.trials ?? baseline.trials?.length;
    $('original-solver-note').append(element('span','eyebrow','ORIGINAL STOCHASTIC SOLVER / MEASURED BASELINE'));
    const stats = element('div','solver-stats');for (const [value,label] of [[`${solved ?? '—'}/${total ?? '—'}`,'full puzzles solved'],[number(summary.mean_slides) === null ? '—' : Number(summary.mean_slides).toFixed(1),'mean slide count'],[number(summary.mean_seconds) === null ? '—' : `${Number(summary.mean_seconds).toFixed(2)} s`,'mean wall time']]) {const stat = element('div');stat.append(element('b','',value),element('small','',label));stats.append(stat);}$('original-solver-note').append(stats);
    const range = number(summary.best_slides) !== null && number(summary.worst_slides) !== null ? `${summary.best_slides}–${summary.worst_slides} slides. ` : '';
    $('original-solver-note').append(element('p','',`${total ?? 'Recorded'} seeded runs use the original solver on its full board, independently replay-validated. ${range}This small solver sample has no model-inference cost. Model rows above are one frozen decision in two option orders; model replays are compact fixtures. These scopes do not support a claim that a model beats the original solver.`));
  } else $('original-solver-note').append(element('p','','The original solver baseline is awaiting its captured source artifact. Model policy agreement and full-puzzle completion remain separate measurements.'));
}
function renderCharts() {
  $('chart-title').textContent = `${games[state.game].name} / agreement & latency`;
  const chart = $('benchmark-chart'); chart.replaceChildren();
  const models = rowsSorted().filter(model => number(metric(model)?.agreement_pct) !== null);
  if (!models.length) {chart.append(element('div','chart-empty',`No measured ${games[state.game].name} results yet. The chart will display saved inference after it is published.`)); $('outcome-note').textContent = '';return;}
  for (const [key,label,formatter,className] of [['agreement_pct','POLICY AGREEMENT / HIGHER IS BETTER',pct,'agreement'],['median_ms','MEDIAN DECISION LATENCY / LOWER IS FASTER',ms,'latency']]) {
    const panel = element('div','chart-panel');panel.append(element('h4','',label)); const max = key === 'agreement_pct' ? 100 : Math.max(1,...models.map(model => number(metric(model)?.[key]) ?? 0));
    for (const model of models) {const value = number(metric(model)?.[key]); const row = element('div',`chart-row ${className}`); row.append(element('span','',model.label || model.name)); const track = element('div','chart-track'); const bar = element('div','chart-bar');bar.style.width = `${value === null ? 0 : Math.max(0,value)/max*100}%`;if (!value) bar.style.minWidth = '0';track.append(bar);row.append(track,element('span','chart-value',formatter(value)));row.setAttribute('aria-label',`${model.label || model.name}: ${formatter(value)}`);panel.append(row);}
    chart.append(panel);
  }
  const best = models[0], summary = metric(best);const outcomes = summary.outcomes || summary.games || [];
  $('outcome-note').replaceChildren();
  const note = element('p');note.append(element('b','',`${best.label || best.name}: `),document.createTextNode(`${summary.valid ?? '—'}/${summary.trials ?? '—'} valid test decisions · P95 ${ms(summary.p95_ms)}.${outcomes.length ? ` ${outcomes.length} saved game outcome${outcomes.length === 1 ? '' : 's'}; inspect the trace for the full context.` : ''}`));$('outcome-note').append(note);
}
function renderMethod() {
  const summary = state.game === 'tetris' ? state.tetrisSummary : state.summary;
  const method = summary?.game_methods?.[state.game] || summary?.methods?.[state.game] || summary?.method;
  const policy = state.models.map(model => metric(model)?.policy).find(Boolean);
  $('method-text').textContent = `${typeof method === 'string' ? method : method ? JSON.stringify(method,null,2) : 'The game engine, benchmark runner and complete traces are published in the source repository. Shared frozen-state evaluation and seeded full games are separate measurements.'}${policy ? ` Current game policy: ${policy}` : ''}`;
  const cache = state.tetrisSummary?.cache_analysis;
  $('cache-note').textContent = cache ? `CLM uses an embedding cache. Fresh and repeated input timing must be distinguished; the complete cache record is included with the summary. ${typeof cache === 'string' ? cache : ''}` : 'Cache effects are possible, including repeated options and states. Runtime adapters and model precisions differ. Check the captured metadata and source before treating latency as a model-only comparison.';
}

function startHuman() {
  state.humanGeneration++;
  const rawSeed = Number($('human-seed').value); const seed = Number.isFinite(rawSeed) ? Math.max(0,Math.min(4294967295,Math.floor(rawSeed))) : 42; $('human-seed').value = String(seed);
  state.human = state.game === '2048' ? new Game2048(seed) : new Connect4();
  $('human-seed').disabled = state.game === 'connect4';
  $('human-heading').textContent = state.game === '2048' ? 'Your move. Make it count.' : 'You are green. Four in a row wins.';
  $('human-explanation').textContent = state.game === '2048' ? 'Slide the board to merge equal tiles. Each tile merges once per move; a valid move spawns a new tile. A seeded browser generator makes restarts repeatable.' : 'Play first against a deterministic depth-4 minimax baseline. It searches four plies and uses center-first tie breaking. This browser opponent is not one of the measured models.';
  $('board-instructions').textContent = state.game === '2048' ? 'Arrow keys / WASD, swipe the board, or use the direction buttons.' : 'Choose a column below the board, or press a number key from 1 to 7.';
  $('context-title').textContent = 'Human play is a separate sandbox.';
  $('context-copy').textContent = state.game === '2048' ? 'Your score is computed in the browser. The model leaderboard uses independently recorded local inference and its published test policy.' : 'The baseline is a small deterministic search algorithm. Its choices and response time are excluded from the model benchmark.';
  $('c4-columns').replaceChildren();
  for (let column = 0; column < 7; column++) {const button = element('button','',String(column+1));button.setAttribute('aria-label',`Drop in column ${column+1}`);button.addEventListener('click',() => humanConnect4(column));$('c4-columns').append(button);}
  renderHuman('Start a fresh board.');
}
function renderHuman(detail = '') {
  if (state.mode !== 'human') return;
  const snapshot = state.human.snapshot();drawBoard(snapshot,true);snapshotStats(snapshot);
  $('board-title').textContent = `${games[state.game].name.toUpperCase()} / ${state.game === '2048' ? `SEED ${state.human.seed}` : 'YOU VS DEPTH-4'}`;
  $('move-counter').textContent = `${snapshot.turn} moves`;
  if (state.game === '2048') {
    $('decision-title').textContent = snapshot.status === 'game_over' ? 'No more moves.' : snapshot.max_tile >= 2048 ? '2048 reached. Keep going.' : 'Space is your most useful tile.';
    $('decision-detail').textContent = detail || 'Choose a direction to slide the board.';
    $('game-status').textContent = snapshot.status === 'game_over' ? `Game over · final score ${snapshot.score}. Restart with the same seed to try again.` : 'Browser game engine · no model inference.';
  } else {
    const status = snapshot.status;
    $('decision-title').textContent = status === 'won' ? snapshot.winner === 1 ? 'You connected four.' : 'The baseline connected four.' : status === 'draw' ? 'A full board. A draw.' : snapshot.player === 2 ? 'The baseline is thinking…' : 'Your turn. Choose a column.';
    $('decision-detail').textContent = detail;
    $('game-status').textContent = `You: green · depth-4 baseline: amber · ${status === 'playing' ? 'browser search, no model inference' : humanize(status)}`;
    const legal = legalColumns(snapshot.board);
    [...$('c4-columns').children].forEach((button,column) => button.disabled = status !== 'playing' || snapshot.player !== 1 || !legal.includes(column));
  }
}
function human2048(direction) {if (state.mode !== 'human' || state.game !== '2048') return; const before = state.human.score;const moved = state.human.move(direction);renderHuman(moved ? `Moved ${direction}${state.human.score > before ? ` · +${state.human.score - before} points` : ''}. A new tile joined the board.` : 'That direction does not change the board. Try another move.');}
function humanConnect4(column) {
  if (state.mode !== 'human' || state.game !== 'connect4' || state.human.player !== 1) return;
  if (!state.human.move(column)) return;
  renderHuman(`You chose column ${column+1}.`);
  if (state.human.status !== 'playing') return;
  const generation = state.humanGeneration,game = state.human;
  setTimeout(() => {
    if (state.mode !== 'human' || state.game !== 'connect4' || state.humanGeneration !== generation || game !== state.human) return;
    const choice = chooseConnect4(game.board,2,4); if (choice !== null) game.move(choice);
    renderHuman(`You chose column ${column+1}. The depth-4 baseline chose column ${choice === null ? '—' : choice+1}.`);
  },120);
}

document.querySelectorAll('.game-tab').forEach(tab => {
  tab.addEventListener('click',() => selectGame(tab.dataset.game));
  tab.addEventListener('keydown',event => {if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return; event.preventDefault();const tabs = [...document.querySelectorAll('.game-tab')];const current = tabs.indexOf(tab);const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length-1 : (current + (event.key === 'ArrowLeft' ? -1 : 1) + tabs.length) % tabs.length;selectGame(tabs[next].dataset.game);tabs[next].focus();});
});
$('mode-replay').addEventListener('click',() => setMode('replay'));$('mode-human').addEventListener('click',() => setMode('human'));
$('model-select').addEventListener('change',() => {state.selectedModel = $('model-select').value;populateRuns();});$('run-select').addEventListener('change',chooseRun);
$('play-button').addEventListener('click',togglePlayback);$('step-button').addEventListener('click',() => {stopPlayback();nextMove();});$('restart-button').addEventListener('click',() => {stopPlayback();state.index = 0;renderReplay();});
$('new-game').addEventListener('click',startHuman);$('human-seed').addEventListener('change',startHuman);
document.querySelectorAll('[data-direction]').forEach(button => button.addEventListener('click',() => human2048(button.dataset.direction)));
document.addEventListener('keydown',event => {
  if (state.mode !== 'human' || event.altKey || event.metaKey || event.ctrlKey || ['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName) || event.target.closest('[role="tablist"]')) return;
  if (state.game === '2048') {const direction = {ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',w:'up',a:'left',s:'down',d:'right'}[event.key];if (direction) {event.preventDefault();human2048(direction);}}
  if (state.game === 'connect4' && /^[1-7]$/.test(event.key)) {event.preventDefault();humanConnect4(Number(event.key)-1);}
});
let touchStart = null;
$('board').addEventListener('pointerdown',event => {if (state.mode === 'human' && state.game === '2048') {touchStart = {x:event.clientX,y:event.clientY,id:event.pointerId};$('board').setPointerCapture(event.pointerId);}});
$('board').addEventListener('pointerup',event => {if (!touchStart || event.pointerId !== touchStart.id) return;const dx = event.clientX-touchStart.x,dy = event.clientY-touchStart.y;touchStart = null;if (Math.max(Math.abs(dx),Math.abs(dy)) < 25) return;human2048(Math.abs(dx)>Math.abs(dy) ? dx>0 ? 'right' : 'left' : dy>0 ? 'down' : 'up');});
$('board').addEventListener('pointercancel',() => {touchStart = null;});
document.addEventListener('visibilitychange',() => {if (document.hidden) stopPlayback();});
selectGame('tetris'); loadData().catch(error => {$('data-status').textContent = `Evidence could not be loaded: ${error.message}. Browser games are available.`;$('data-status').classList.add('error');console.error(error);});
