"""Click through the shipped evaluation set: three instances per scenario.

reviewer's request of 04.09.: before the 1,900-instance package goes to the
collaborators, look at three examples of every one of the 106 scenarios — ten
single-flaw categories, 36 pairs, 60 triples — with the ground truth beside the
flawed chart, the injected edits, the summary and the table. Space bar advances,
1 and 0 record a verdict, the export button writes them out.

Reads the built release, so what is shown is exactly what ships.

    .venv/bin/python src/review_release.py --per 3 \
        --release release/chartrepairbench-v1.3-eval \
        --out data/v0/review_release_v13.html
"""
import argparse
import csv
import hashlib
import html
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

CSS = """
html,body{margin:0;background:#f6f5f1;color:#1d1c1a;font:14px/1.45 -apple-system,"Helvetica Neue",Arial,sans-serif}
#top{display:flex;gap:16px;align-items:baseline;padding:9px 18px;background:#e9e6dd;border-bottom:1px solid #d3cfc3;position:sticky;top:0;z-index:5}
#top b{font-size:15px}
#top .muted{color:#5d5a54}
#top .tag{background:#dfd9cb;border-radius:9px;padding:1px 9px;font-size:12px}
#top button{margin-left:auto;font:inherit;padding:4px 10px;background:#fff;border:1px solid #b8b3a6;cursor:pointer}
#help{padding:6px 18px 10px;color:#5d5a54;font-size:12px}
#main{display:grid;grid-template-columns:1fr 1fr;gap:13px;padding:0 18px 40px}
.chart{background:#fff;border:1px solid #d3cfc3;padding:8px}
.chart img{width:100%;display:block;background:#fff}
.cap{font-size:12px;color:#5d5a54;padding-top:5px}
.gt .cap::before{content:"ground truth  ";font-weight:600;color:#2f6b2f}
.fl .cap::before{content:"flawed  ";font-weight:600;color:#9b2f2f}
.panel{background:#fff;border:1px solid #d3cfc3;padding:10px 12px;overflow:auto;max-height:40vh}
.panel h3{margin:0 0 6px;font-size:13px;color:#5d5a54;font-weight:600}
.flaws li{margin:5px 0}
.flaws b{color:#9b2f2f}
table.t{border-collapse:collapse;font-size:12px}
table.t td,table.t th{border:1px solid #ddd8cb;padding:2px 7px;text-align:left;white-space:nowrap}
table.t td{white-space:pre-wrap;min-width:65px}
.verdict{font-weight:600}.ok{color:#2f6b2f}.bad{color:#9b2f2f}
"""

JS = """
const {items,review}=JSON.parse(document.getElementById('review-data').textContent);
let i=0;const key='crb_release_review:'+review.dataset_sha256+':'+review.selection_sha256;
let v={};try{const saved=JSON.parse(localStorage.getItem(key)||'{}');
 if(saved.dataset_sha256===review.dataset_sha256&&saved.selection_sha256===review.selection_sha256)
  for(const it of items)if(saved.verdicts?.[it.id]===0||saved.verdicts?.[it.id]===1)v[it.id]=saved.verdicts[it.id];
}catch(e){v={}}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function csvTable(rows){
 if(!rows.length)return '';let h='<table class="t"><tr>'+rows[0].map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr>';
 for(const r of rows.slice(1))h+='<tr>'+r.map(c=>'<td>'+esc(c)+'</td>').join('')+'</tr>';return h+'</table>'}
function render(){const it=items[i];const d=v[it.id];const done=Object.keys(v).length;
 document.getElementById('top').innerHTML='<b>'+(i+1)+' / '+items.length+'</b>'+
  '<span class="tag">'+esc(it.arity)+'</span><span>'+esc(it.combination)+'</span>'+
  '<span class="muted">'+esc(it.library)+' · '+esc(it.chart_type)+' · Gruppe '+esc(it.group)+'</span>'+
  '<span class="verdict '+(d===1?'ok':d===0?'bad':'')+'">'+(d===1?'passt':d===0?'passt nicht':'')+'</span>'+
  '<span class="muted">'+done+' beurteilt</span>'+
  '<button onclick="exportJson()">Export</button>';
 document.getElementById('main').innerHTML=
  '<div class="chart gt"><img src="'+it.gt+'"><div class="cap">'+esc(it.id)+'</div></div>'+
  '<div class="chart fl"><img src="'+it.fl+'"><div class="cap">'+esc(it.combination)+'</div></div>'+
  '<div class="panel"><h3>Injizierte Flaws ('+it.flaws.length+')</h3><ul class="flaws">'+
   it.flaws.map(f=>'<li><b>'+esc(f.category)+'</b> '+esc(f.description)+'</li>').join('')+
  '</ul><h3>summary.txt</h3><div>'+esc(it.summary)+'</div></div>'+
  '<div class="panel"><h3>table.csv (vollständig)</h3>'+csvTable(it.table_rows)+'</div>';
 window.scrollTo(0,0)}
function reviewRecord(){return {...review,verdicts:v}}
function exportJson(){const b=new Blob([JSON.stringify(reviewRecord(),null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='release_review_verdicts.json';a.click()}
document.addEventListener('keydown',e=>{
 if(e.code==='Space'||e.key==='ArrowRight'){e.preventDefault();if(i<items.length-1){i++;render()}}
 else if(e.key==='Backspace'||e.key==='ArrowLeft'){e.preventDefault();if(i>0){i--;render()}}
 else if(e.key==='1'||e.key==='0'){v[items[i].id]=e.key==='1'?1:0;try{localStorage.setItem(key,JSON.stringify(reviewRecord()))}catch(err){};render()}
 else if(e.key==='g'){const n=parseInt(prompt('Gehe zu Nummer (1-'+items.length+')'),10);if(n>=1&&n<=items.length){i=n-1;render()}}});
render();
"""


def build_review(release: Path, per: int, image_base: Path):
    manifest = (release / "dataset.json").read_bytes()
    data = json.loads(manifest)
    by_scenario = defaultdict(list)
    for item in data["items"]:
        by_scenario[(item["arity"], item["combination"])].append(item)

    rng = random.Random(SEED)
    order = {"single": 0, "pair": 1, "triple": 2}
    picks = []
    for key in sorted(by_scenario, key=lambda k: (order[k[0]], k[1])):
        bucket = sorted(by_scenario[key], key=lambda i: i["id"])
        rng.shuffle(bucket)
        picks.extend(bucket[:per])

    # The page lives next to the release, so the image paths stay relative.
    rows, selected_files = [], []
    for item in picks:
        with (release / item["files"]["table"]).open(
                encoding="utf-8", errors="replace", newline="") as stream:
            table_rows = list(csv.reader(stream))
        rows.append({
            "id": item["id"], "arity": item["arity"], "combination": item["combination"],
            "library": item["library"], "chart_type": item["chart_type"],
            "group": item["group"], "flaws": item["flaws"],
            "gt": str(image_base / item["files"]["ground_truth_render"]),
            "fl": str(image_base / item["files"]["flawed_render"]),
            "summary": (release / item["files"]["summary"]).read_text(
                encoding="utf-8", errors="replace"),
            "table_rows": table_rows,
        })
        selected_files.append({"id": item["id"], "files": {
            name: hashlib.sha256((release / path).read_bytes()).hexdigest()
            for name, path in item["files"].items()}})

    # Bind verdicts to the selected artifacts too: a render can change without
    # changing its path in dataset.json.
    selection = {"seed": SEED, "per_scenario": per, "items": selected_files}
    review = {
        "schema": 2, "dataset": data["meta"]["name"],
        "dataset_sha256": hashlib.sha256(manifest).hexdigest(),
        "selection_sha256": hashlib.sha256(json.dumps(
            selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "seed": SEED, "per_scenario": per,
        "selected_instance_ids": [row["id"] for row in rows],
    }
    return {"items": rows, "review": review}, len(by_scenario)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", type=Path, default=ROOT / "release/chartrepairbench-v1.3-eval")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data/v0/review_release_v13.html")
    args = ap.parse_args(argv)
    if args.per < 1:
        ap.error("--per must be at least 1")
    image_base = Path(os.path.relpath(args.release.resolve(), args.out.parent.resolve()))
    payload, scenarios = build_review(args.release, args.per, image_base)
    rows, review = payload["items"], payload["review"]
    # JSON embedded in HTML must not let CSV or summary content close the script.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")

    page = ("<!doctype html><meta charset='utf-8'><title>ChartRepairBench release review</title>"
            f"<style>{CSS}</style><div id='top'></div>"
            "<div id='help'>Leertaste oder → weiter · ← zurück · 1 = passt · 0 = passt nicht · "
            "g = gehe zu Nummer · Export sichert die Urteile. "
            f"{args.per} Instanzen je Szenario, {scenarios} Szenarien.<br>"
            f"Quelle: {html.escape(review['dataset'])} · Dataset SHA-256: "
            f"{review['dataset_sha256']} · Auswahl SHA-256: {review['selection_sha256']}"
            "</div><div id='main'></div>"
            f"<script id='review-data' type='application/json'>{encoded}</script><script>{JS}</script>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    counts = defaultdict(int)
    for row in rows:
        counts[row["arity"]] += 1
    print(f"wrote {args.out}: {len(rows)} instances ({dict(counts)}) from "
          f"{scenarios} scenarios")


if __name__ == "__main__":
    main()
