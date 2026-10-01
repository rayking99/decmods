const $ = id => document.getElementById(id);
const palette = {I:'#74d2d9',O:'#eac976',T:'#b89ae6',S:'#a2cd82',Z:'#e08b87',J:'#81a5dc',L:'#e4ad79'};
const shapes = {I:[[0,0],[1,0],[2,0],[3,0]],O:[[0,0],[1,0],[0,1],[1,1]],T:[[1,0],[0,1],[1,1],[2,1]],S:[[1,0],[2,0],[0,1],[1,1]],Z:[[0,0],[1,0],[1,1],[2,1]],J:[[0,0],[0,1],[1,1],[2,1]],L:[[2,0],[0,1],[1,1],[2,1]]};
let state = null, animationStart = 0, pendingKey = '', oldPieces = 0, clearFlash = 0, oldLines = 0;
const ctx = $('board').getContext('2d'), nextCtx = $('next').getContext('2d');
const fmtMs = ms => ms == null ? '—' : ms >= 1000 ? `${(ms/1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
function block(context,x,y,color,size=32,ghost=false){
  const inset=2;
  if(ghost){context.strokeStyle=color;context.lineWidth=1.5;context.globalAlpha=.6;context.strokeRect(x*size+inset,y*size+inset,size-4,size-4);context.globalAlpha=1;return;}
  context.fillStyle=color;context.beginPath();context.roundRect(x*size+inset,y*size+inset,size-4,size-4,3);context.fill();
  context.fillStyle='#ffffff22';context.fillRect(x*size+5,y*size+4,size-10,2);
}
function draw(){
  ctx.fillStyle='#0d1315';ctx.fillRect(0,0,320,640);ctx.strokeStyle='#1e292a';ctx.lineWidth=.5;
  for(let x=0;x<=10;x++){ctx.beginPath();ctx.moveTo(x*32,0);ctx.lineTo(x*32,640);ctx.stroke();}
  for(let y=0;y<=20;y++){ctx.beginPath();ctx.moveTo(0,y*32);ctx.lineTo(320,y*32);ctx.stroke();}
  if(state){
    state.board.forEach((row,y)=>row.forEach((v,x)=>{if(v)block(ctx,x,y,palette[v]);}));
    if(state.pending){const m=state.pending;const elapsed=performance.now()-animationStart;const duration=Math.max(100,Math.min(state.delay*1000-60,450));const t=Math.min(1,elapsed/duration);const offset=(1-t*t)*m.y;
      m.cells.forEach(([x,y])=>block(ctx,x,y,palette[m.piece],32,true));
      m.cells.forEach(([x,y])=>block(ctx,x,y-offset,palette[m.piece]));
    }else if(!['game_over','complete','error'].includes(state.phase)){
      const shape=shapes[state.current];shape.forEach(([x,y])=>block(ctx,x+3,y+.2,palette[state.current],32,true));
    }
    if(performance.now()-clearFlash<350){ctx.fillStyle=`rgba(196,237,155,${.15*(1-(performance.now()-clearFlash)/350)})`;ctx.fillRect(0,0,320,640);}
  }
  requestAnimationFrame(draw);
}
function drawNext(){nextCtx.clearRect(0,0,100,170);state.next.slice(0,3).forEach((p,i)=>shapes[p].forEach(([x,y])=>block(nextCtx,x+.25,y+i*3,palette[p],18)));}
function render(s){
  if(!state||state.epoch!==s.epoch){oldPieces=0;oldLines=0;pendingKey='';}
  if(s.lines>oldLines)clearFlash=performance.now();oldLines=s.lines;oldPieces=s.pieces;
  const key=s.pending?`${s.epoch}:${s.pieces}:${s.pending.id}`:'';
  if(key!==pendingKey){pendingKey=key;animationStart=performance.now();}
  state=s;drawNext();
  $('lines').textContent=s.lines;$('pieces').textContent=s.pieces;$('score').textContent=s.score;
  $('holes').textContent=s.metrics.holes;$('height').textContent=s.metrics.max_height;$('seed-label').textContent=`SEED ${s.seed}`;
  const modelName=s.model.name.replace(':latest','');
  const labels={ready:'Ready when you are',thinking:s.running?`${modelName} is choosing…`:'Finishing decision · paused',placing:s.running?'Placing the chosen piece':'Move ready · paused',running:'Playing live',paused:'Paused',game_over:'Game over',complete:'Run complete',error:'Decision stopped'};
  $('phase').textContent=labels[s.phase]||s.phase;document.body.classList.toggle('thinking',s.phase==='thinking');
  $('toggle').textContent=s.running?'Pause Ⅱ':`Let ${modelName} play ↗`;
  $('toggle').disabled=['game_over','complete'].includes(s.phase);$('step').disabled=s.running||['game_over','complete'].includes(s.phase);
  $('mean-latency').textContent=fmtMs(s.mean_latency_ms);
  $('model-label').textContent=`${modelName} · ${s.model.provider||'Ollama'}`;
  $('status-dot').style.background=s.error?'#e08b87':'#c4ed9b';
  $('overlay').hidden=!['game_over','complete'].includes(s.phase);
  $('overlay-title').textContent=s.phase==='complete'?'Run complete':'Game over';
  $('overlay-detail').textContent=`${s.pieces} pieces · ${s.lines} lines cleared\nRestart to try the same sequence.`;
  $('error').hidden=!s.error;$('error').textContent=s.error||'';
  const d=s.latest;
  if(d){
    const m=d.options.find(x=>x.id===d.choice);$('chosen-piece').textContent=m.piece;$('chosen-piece').style.color=palette[m.piece];
    $('choice-title').textContent=`Column ${m.x+1} · ${m.rotation}°`;
    $('choice-detail').textContent=`${m.cleared} rows cleared · ${m.metrics.holes} holes · height ${m.metrics.max_height}`;
    $('source-label').textContent=d.source==='forced'?'Only legal move':`Chosen by ${modelName}`;
    $('latency').textContent=fmtMs(d.latency_ms);$('candidate-count').textContent=d.candidate_count;
    $('rounds').textContent=d.calls.length>1?'Final round · 2 group winners':'All legal landings';
    const options=d.options.slice().sort((a,b)=>(d.probabilities?.[b.id]||0)-(d.probabilities?.[a.id]||0));
    $('options').innerHTML=options.slice(0,5).map(o=>{const p=d.probabilities?.[o.id]??1;return `<div class="option ${o.id===d.choice?'selected':''}"><div class="option-line"><span>Column ${o.x+1} · ${o.rotation}° <small>${o.cleared} rows / ${o.metrics.holes} holes</small></span><span>${(p*100).toFixed(1)}%</span></div><div class="bar-track"><div class="bar" style="width:${p*100}%"></div></div></div>`;}).join('');
    $('prob-note').textContent=`${d.calls.length} model call${d.calls.length===1?'':'s'} · Preference concentration ${((d.concentration??0)*100).toFixed(0)}%. This is not a correctness probability.`;
  }else{
    $('chosen-piece').textContent='—';$('choice-title').textContent='Ready to decide';$('choice-detail').textContent=`Every legal landing is offered to ${modelName}.`;$('source-label').textContent='Awaiting model';
    $('latency').textContent='—';$('candidate-count').textContent='—';$('rounds').textContent='Selected probability';$('options').innerHTML='<div class="empty">Once play starts, actual probabilities appear here.</div>';
    $('prob-note').textContent='Probabilities express model preference, not the chance that a move is correct.';
  }
  $('history').innerHTML=s.history.length?s.history.slice().reverse().map(h=>`<div class="history-row"><span>#${h.turn}</span><b style="color:${palette[h.move.piece]}">${h.move.piece}</b><span>col ${h.move.x+1} / ${h.move.rotation}°</span><span class="${h.move.cleared?'cleared':''}">${h.move.cleared?'+'+h.move.cleared+' lines':'—'}</span><span class="elapsed">${fmtMs(h.decision.latency_ms)}</span></div>`).join(''):'<div class="empty">Each move leaves a trace.</div>';
}
async function command(action,data={}){try{const response=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const result=await response.json();if(!response.ok)throw Error(result.error);render(result);}catch(e){$('error').hidden=false;$('error').textContent=e.message;}}
$('toggle').onclick=()=>command(state?.running?'pause':'start');$('step').onclick=()=>command('step');$('reset').onclick=()=>command('reset');
$('download').onclick=async e=>{e.preventDefault();const response=await fetch('/api/replay');const blob=new Blob([JSON.stringify(await response.json(),null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`decision-tetris-${state.run_id}.json`;a.click();URL.revokeObjectURL(url);};
async function poll(){try{const r=await fetch('/api/state');if(!r.ok)throw Error(`Server HTTP ${r.status}`);render(await r.json());}catch(e){$('phase').textContent='Server disconnected';$('error').hidden=false;$('error').textContent=e.message;}finally{setTimeout(poll,150);}}
draw();poll();
