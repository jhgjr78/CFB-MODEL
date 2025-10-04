async function loadJSON(url){ const r = await fetch(url, {cache:"no-store"}); if(!r.ok) throw new Error(r.status); return r.json(); }
function fmt(v){ return v==null ? "—" : (typeof v==="number" ? (Math.abs(v)%1?v.toFixed(1):v.toString()) : v); }

function rowHTML(r){
  const oursSpread = r.adj_spread, vegasSpread = r.vegas_spread;
  const oursTotal  = r.total_pts,  vegasTotal  = r.vegas_total;
  const sDelta = (vegasSpread==null)? "": ( (oursSpread - vegasSpread) );
  const tDelta = (vegasTotal==null)?  "": ( (oursTotal - vegasTotal) );

  const sTag = (sDelta==="")?"": `<span class="${sDelta>=0?'deltaUp':'deltaDn'}">${sDelta>=0?'+':''}${fmt(sDelta)}</span>`;
  const tTag = (tDelta==="")?"": `<span class="${tDelta>=0?'deltaUp':'deltaDn'}">${tDelta>=0?'+':''}${fmt(tDelta)}</span>`;

  return `
  <tr class="row">
    <td>${r.away} @ <span class="fav">${r.home}</span></td>
    <td class="col-ours">Fav: ${r.favored}<br/>Spread: ${fmt(oursSpread)}<br/>Total: ${fmt(oursTotal)}</td>
    <td class="col-vegas">V Spread: ${fmt(vegasSpread)}<br/>V Total: ${fmt(vegasTotal)}</td>
    <td>Δ Spread: ${sTag}<br/>Δ Total: ${tTag}</td>
    <td>FP ${fmt(r.fp)} | HY ${fmt(r.hidden)} | XPL ${fmt(r.xpl)} | SR ${fmt(r.sr)} | HV ${fmt(r.havoc)} | REC ${fmt(r.recency)}</td>
    <td>Plays ~ ${fmt(r.plays_est)}</td>
  </tr>`;
}

async function render(){
  const meta = document.getElementById('meta');
  const holder = document.getElementById('table');
  holder.innerHTML = '<div class="hint">Loading…</div>';
  try{
    const rows = await loadJSON('week_preds.json');
    meta.textContent = `${rows.length} games`;
    if(!rows.length){ holder.innerHTML = '<div class="hint">No predictions yet. Run “Build Dataset” then “Weekly Runner”.</div>'; return; }
    const head = `
      <table>
        <thead><tr>
          <th>Game</th><th>Our Model</th><th>Vegas</th><th>Δ</th><th>Factors</th><th>Pace</th>
        </tr></thead>
        <tbody>${rows.map(rowHTML).join('')}</tbody>
      </table>`;
    holder.innerHTML = head;
  } catch(e){
    holder.innerHTML = `<div class="hint">Could not load week_preds.json (${e}).</div>`;
  }
}

document.getElementById('refresh').addEventListener('click', render);
render();