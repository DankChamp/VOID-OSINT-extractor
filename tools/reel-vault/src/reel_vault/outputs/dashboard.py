"""dashboard.html — self-contained, searchable/filterable HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reel_vault.storage.db import Store
from reel_vault.storage.paths import WorkspacePaths

CSS = """
  input[type="search"] {{
    padding: 0.4rem 0.6rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    flex: 1;
    min-width: 200px;
  }}
  select {{
    padding: 0.4rem 0.6rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }}
  button {{
    padding: 0.4rem 0.8rem;
    border: 1px solid #333;
    background: #fff;
    border-radius: 4px;
    cursor: pointer;
  }}
  button.active {{
    background: #007acc;
    color: #fff;
    border-color: #007acc;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  th {{
    background: #f0f0f0;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    border-bottom: 1px solid #ddd;
  }}
  td {{
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #eee;
    font-size: 0.9rem;
  }}
  tr:hover {{
    background: #fafafa;
  }}
  tr.archived {{
    border-left: 4px solid #2ecc71;
  }}
  tr.stub {{
    border-left: 4px solid #e74c3c;
  }}
  tr.extracted {{
    border-left: 4px solid #3498db;
  }}
  .badge {{
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.75rem;
    background: #e0e0e0;
  }}
  .badge.extract {{
    background: #d1ecf1;
    color: #0c5460;
  }}
  .badge.archive {{
    background: #d4edda;
    color: #155724;
  }}
  .badge.stub {{
    background: #f8d7da;
    color: #721c24;
  }}
  .empty {{
    padding: 2rem;
    text-align: center;
    color: #666;
  }}
  a {{
    color: #007acc;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .count {{
    font-weight: bold;
    color: #007acc;
  }}
"""


def render_dashboard_html(
    store: Store,
    paths: WorkspacePaths,
) -> Path:
    """Write ``outputs/dashboard.html`` — a self-contained, client-side filtered dashboard."""
    out_dir = paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard.html"

    reels = store.list_reels(limit=10000)

    records: list[dict[str, Any]] = []
    for reel in reels:
        shortcode = str(reel["shortcode"])
        classification = store.get_classification(shortcode)
        triage = store.get_triage(shortcode)

        caption = ""
        if triage and triage.get("reason"):
            caption = str(triage["reason"])
        elif (classification or {}).get("primary"):
            caption = str((classification or {}).get("primary"))

        records.append(
            {
                "shortcode": shortcode,
                "url": reel.get("url") or "",
                "author": reel.get("author") or "",
                "status": reel.get("status") or "",
                "decision": (triage or {}).get("decision") or "",
                "score": (triage or {}).get("value_score") or 0,
                "category": (classification or {}).get("primary") or "",
                "saved": (reel.get("saved_at") or "")[:10],
                "caption": caption,
            }
        )

    records_json = json.dumps(records, ensure_ascii=False)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>reel-vault dashboard</title>
<style>
*{{
  box-sizing: border-box;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 1rem;
  background: #f5f5f5;
}}
h1 {{
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}}
.toolbar {{
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  align-items: center;
}}
input[type="search"] {{
  padding: 0.4rem 0.6rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  flex: 1;
  min-width: 200px;
}}
select {{
  padding: 0.4rem 0.6rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}}
button {{
  padding: 0.4rem 0.8rem;
  border: 1px solid #333;
  background: #fff;
  border-radius: 4px;
  cursor: pointer;
}}
button.active {{
  background: #007acc;
  color: #fff;
  border-color: #007acc;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
th {{
  background: #f0f0f0;
  padding: 0.6rem 0.8rem;
  text-align: left;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  border-bottom: 1px solid #ddd;
}}
td {{
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid #eee;
  font-size: 0.9rem;
}}
tr:hover {{
  background: #fafafa;
}}
tr.archived {{
  border-left: 4px solid #2ecc71;
}}
tr.stub {{
  border-left: 4px solid #e74c3c;
}}
tr.extracted {{
  border-left: 4px solid #3498db;
}}
.badge {{
  display: inline-block;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-size: 0.75rem;
  background: #e0e0e0;
}}
.badge.extract {{
  background: #d1ecf1;
  color: #0c5460;
}}
.badge.archive {{
  background: #d4edda;
  color: #155724;
}}
.badge.stub {{
  background: #f8d7da;
  color: #721c24;
}}
.empty {{
  padding: 2rem;
  text-align: center;
  color: #666;
}}
a {{
  color: #007acc;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
.count {{
  font-weight: bold;
  color: #007acc;
}}
</style>
</head>
<body>
<h1>reel-vault dashboard</h1>
<div class="toolbar">
  <input type="search" id="search" placeholder="Search shortcode, author, caption…">
  <select id="filter-status">
    <option value="all">All statuses</option>
  </select>
  <select id="filter-decision">
    <option value="all">All decisions</option>
  </select>
  <select id="filter-category">
    <option value="all">All categories</option>
  </select>
  <span class="count" id="count">0 reels</span>
</div>
<table>
  <thead>
    <tr>
      <th>Shortcode</th>
      <th>Author</th>
      <th>Status</th>
      <th>Decision</th>
      <th>Score</th>
      <th>Category</th>
      <th>Saved</th>
      <th>Caption / reason</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<script>
const records = {records_json};

const statuses = [...new Set(records.map(r => r.status))];
const decisions = [...new Set(records.map(r => r.decision))];
const categories = [...new Set(records.map(r => r.category))];

const statusSel = document.getElementById('filter-status');
const decisionSel = document.getElementById('filter-decision');
const categorySel = document.getElementById('filter-category');
const searchInput = document.getElementById('search');
const tbody = document.getElementById('tbody');
const countEl = document.getElementById('count');

statuses.forEach((s) => {{
  const o = document.createElement('option');
  o.value = s;
  o.textContent = s;
  statusSel.appendChild(o);
}});
decisions.forEach((d) => {{
  const o = document.createElement('option');
  o.value = d;
  o.textContent = d;
  decisionSel.appendChild(o);
}});
categories.forEach((c) => {{
  const o = document.createElement('option');
  o.value = c;
  o.textContent = c;
  categorySel.appendChild(o);
}});

function render() {{
  const search = searchInput.value.toLowerCase();
  const status = statusSel.value;
  const decision = decisionSel.value;
  const category = categorySel.value;

  const filtered = records.filter(r => {{
    if (status !== 'all' && r.status !== status) return false;
    if (decision !== 'all' && r.decision !== decision) return false;
    if (category !== 'all' && r.category !== category) return false;
    const haystack = (r.shortcode + ' ' + r.author + ' ' + r.caption).toLowerCase();
    return haystack.includes(search);
  }});

  tbody.innerHTML = '';
  if (filtered.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No matching reels.</td></tr>';
  }} else {{
    for (const r of filtered) {{
      const tr = document.createElement('tr');
      tr.className = r.decision;
      const isExtract = r.decision === 'extract';
      const isArchive = r.decision === 'archive';
      const badgeClass = isExtract ? 'extract' : isArchive ? 'archive' : 'stub';
      tr.innerHTML = `
        <td><a href="${{r.url}}" target="_blank">${{r.shortcode}}</a></td>
        <td>${{r.author}}</td>
        <td>${{r.status}}</td>
        <td><span class="badge ${{badgeClass}}">${{r.decision}}</span></td>
        <td class="score">${{r.score}}</td>
        <td>${{r.category}}</td>
        <td>${{r.saved}}</td>
        <td>${{r.caption}}</td>
      `;
      tbody.appendChild(tr);
    }}
  }}
  countEl.textContent = filtered.length + ' reels';
}}

searchInput.addEventListener('input', render);
statusSel.addEventListener('change', render);
decisionSel.addEventListener('change', render);
categorySel.addEventListener('change', render);
render();
</script>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
