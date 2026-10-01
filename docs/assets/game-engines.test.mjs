import test from 'node:test';
import assert from 'node:assert/strict';
import {slideLine,slide2048,Game2048,canMove2048,Connect4,dropConnect4,winningCells,chooseConnect4} from './game-engines.mjs';

test('2048 tiles merge once per move and add the resulting tile values',() => {
  assert.deepEqual(slideLine([2,2,2,2]),{line:[4,4,0,0],gained:8});
  assert.deepEqual(slideLine([2,2,4,0]),{line:[4,4,0,0],gained:4});
  assert.deepEqual(slideLine([4,0,4,4]),{line:[8,4,0,0],gained:8});
});
test('2048 all four directions preserve the board orientation',() => {
  const board = [[2,0,0,2],[0,0,0,0],[0,0,0,0],[2,0,0,2]];
  assert.deepEqual(slide2048(board,'right').board,[[0,0,0,4],[0,0,0,0],[0,0,0,0],[0,0,0,4]]);
  assert.deepEqual(slide2048(board,'up').board,[[4,0,0,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]]);
  assert.deepEqual(slide2048(board,'down').board,[[0,0,0,0],[0,0,0,0],[0,0,0,0],[4,0,0,4]]);
});
test('2048 invalid moves do not advance the generator or spawn a tile',() => {
  const game = new Game2048(42); game.board = [[2,4,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  const before = game.snapshot(); assert.equal(game.move('left'),false);assert.deepEqual(game.snapshot(),before);
  const first = new Game2048(123),second = new Game2048(123);
  for (const direction of ['left','down','right','up','left','down']) {first.move(direction);second.move(direction);}
  assert.deepEqual(first.snapshot(),second.snapshot());
});
test('2048 full checkerboard ends while a full mergeable board does not',() => {
  assert.equal(canMove2048([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]]),false);
  assert.equal(canMove2048([[2,2,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]]),true);
});
test('Connect Four respects gravity, full columns and terminal wins',() => {
  const game = new Connect4();for (const column of [0,1,0,1,0,1,0]) assert.equal(game.move(column),true);
  assert.equal(game.status,'won');assert.equal(game.winner.player,1);assert.equal(game.move(6),false);
  const full = new Connect4().board;for (let y = 0; y < 6; y++) full[y][0] = y%2+1;
  assert.equal(dropConnect4(full,0,1),null);assert.equal(dropConnect4(full,-1,1),null);
});
test('Connect Four detects both diagonals',() => {
  const left = new Connect4().board;for (let i = 0; i < 4; i++) left[5-i][i] = 1;
  assert.equal(winningCells(left).player,1);
  const right = new Connect4().board;for (let i = 0; i < 4; i++) right[2+i][i] = 2;
  assert.equal(winningCells(right).player,2);
});
test('depth-4 baseline takes an immediate win and blocks an immediate loss',() => {
  const win = new Connect4().board;win[5] = [2,2,2,0,1,1,0];
  assert.equal(chooseConnect4(win,2,4),3);
  const block = new Connect4().board;block[5] = [1,1,1,0,2,0,0];
  assert.equal(chooseConnect4(block,2,4),3);
  assert.equal(chooseConnect4(new Connect4().board,2,4),3);
});
