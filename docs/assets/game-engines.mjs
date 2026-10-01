// Browser games are deterministic human-play sandboxes, separate from model traces.
export function seededRandom(seed = 42) {
  let state = (Number(seed) >>> 0) || 0x6d2b79f5;
  return () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

export function slideLine(line) {
  const compact = line.filter(Boolean), result = [];
  let gained = 0;
  for (let i = 0; i < compact.length; i++) {
    if (compact[i] === compact[i + 1]) {
      const value = compact[i] * 2; result.push(value); gained += value; i++;
    } else result.push(compact[i]);
  }
  while (result.length < line.length) result.push(0);
  return {line: result, gained};
}

export function slide2048(board, direction) {
  if (!['left', 'right', 'up', 'down'].includes(direction)) throw new Error('Unknown direction');
  const next = board.map(row => [...row]);
  let gained = 0;
  for (let i = 0; i < 4; i++) {
    const vertical = direction === 'up' || direction === 'down';
    const reverse = direction === 'right' || direction === 'down';
    let line = Array.from({length: 4}, (_, j) => vertical ? board[j][i] : board[i][j]);
    if (reverse) line.reverse();
    const merged = slideLine(line); gained += merged.gained;
    line = reverse ? merged.line.reverse() : merged.line;
    for (let j = 0; j < 4; j++) if (vertical) next[j][i] = line[j]; else next[i][j] = line[j];
  }
  return {board: next, gained, changed: next.some((row, y) => row.some((value, x) => value !== board[y][x]))};
}

export function canMove2048(board) {
  return ['left', 'right', 'up', 'down'].some(direction => slide2048(board, direction).changed);
}

export class Game2048 {
  constructor(seed = 42) {
    this.seed = Number(seed); this.random = seededRandom(seed);
    this.board = Array.from({length: 4}, () => Array(4).fill(0));
    this.score = 0; this.turn = 0; this.status = 'playing';
    this.spawn(); this.spawn();
  }
  spawn() {
    const empty = this.board.flatMap((row, y) => row.flatMap((value, x) => value ? [] : [[y, x]]));
    if (!empty.length) return;
    const [y, x] = empty[Math.floor(this.random() * empty.length)];
    this.board[y][x] = this.random() < .9 ? 2 : 4;
  }
  move(direction) {
    if (this.status === 'game_over') return false;
    const outcome = slide2048(this.board, direction);
    if (!outcome.changed) return false;
    this.board = outcome.board; this.score += outcome.gained; this.turn++; this.spawn();
    this.status = canMove2048(this.board) ? 'playing' : 'game_over';
    return true;
  }
  snapshot() {
    return {board: this.board.map(row => [...row]), score: this.score, turn: this.turn,
      max_tile: Math.max(...this.board.flat()), status: this.status};
  }
}

export function legalColumns(board) {
  return Array.from({length: 7}, (_, column) => column).filter(column => board[0][column] === 0);
}

export function dropConnect4(board, column, player) {
  if (!Number.isInteger(column) || column < 0 || column > 6 || board[0][column]) return null;
  const next = board.map(row => [...row]);
  for (let row = 5; row >= 0; row--) if (!next[row][column]) {next[row][column] = player; break;}
  return next;
}

export function winningCells(board) {
  for (let row = 0; row < 6; row++) for (let column = 0; column < 7; column++) {
    const player = board[row][column]; if (!player) continue;
    for (const [dy, dx] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      const cells = Array.from({length: 4}, (_, i) => [row + dy * i, column + dx * i]);
      if (cells.every(([y, x]) => y >= 0 && y < 6 && x >= 0 && x < 7 && board[y][x] === player)) return {player, cells};
    }
  }
  return null;
}

const COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6];
function heuristicConnect4(board, player) {
  const opponent = 3 - player;
  let score = board.reduce((total, row) => total + (row[3] === player ? 3 : row[3] === opponent ? -3 : 0), 0);
  for (let y = 0; y < 6; y++) for (let x = 0; x < 7; x++) for (const [dy, dx] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const endY = y + dy * 3, endX = x + dx * 3;
    if (endY < 0 || endY > 5 || endX < 0 || endX > 6) continue;
    const cells = Array.from({length: 4}, (_, i) => board[y + dy * i][x + dx * i]);
    const own = cells.filter(value => value === player).length, other = cells.filter(value => value === opponent).length;
    if (!other) score += [0, 1, 8, 64, 100000][own];
    if (!own) score -= [0, 1, 8, 64, 100000][other];
  }
  return score;
}

export function chooseConnect4(board, player = 2, depth = 4) {
  const rootPlayer = player;
  function search(position, remaining, toMove, alpha, beta) {
    const winner = winningCells(position);
    if (winner) return winner.player === rootPlayer ? 100000 + remaining : -100000 - remaining;
    const legal = legalColumns(position);
    if (!legal.length) return 0;
    if (!remaining) return heuristicConnect4(position, rootPlayer);
    const maximize = toMove === rootPlayer;
    let best = maximize ? -Infinity : Infinity;
    for (const column of COLUMN_ORDER.filter(column => legal.includes(column))) {
      const value = search(dropConnect4(position, column, toMove), remaining - 1, 3 - toMove, alpha, beta);
      best = maximize ? Math.max(best, value) : Math.min(best, value);
      if (maximize) alpha = Math.max(alpha, best); else beta = Math.min(beta, best);
      if (alpha >= beta) break;
    }
    return best;
  }
  let bestColumn = null, best = -Infinity;
  for (const column of COLUMN_ORDER.filter(column => legalColumns(board).includes(column))) {
    const value = search(dropConnect4(board, column, player), Math.max(0, depth - 1), 3 - player, -Infinity, Infinity);
    if (value > best) {best = value; bestColumn = column;}
  }
  return bestColumn;
}

export class Connect4 {
  constructor() {this.board = Array.from({length: 6}, () => Array(7).fill(0)); this.turn = 0; this.player = 1; this.status = 'playing'; this.winner = null;}
  move(column) {
    if (this.status !== 'playing') return false;
    const next = dropConnect4(this.board, column, this.player); if (!next) return false;
    this.board = next; this.turn++;
    this.winner = winningCells(this.board);
    if (this.winner) this.status = 'won';
    else if (!legalColumns(this.board).length) this.status = 'draw';
    else this.player = 3 - this.player;
    return true;
  }
  snapshot() {return {board: this.board.map(row => [...row]), turn: this.turn, player: this.player, winner: this.winner?.player || null, status: this.status};}
}
