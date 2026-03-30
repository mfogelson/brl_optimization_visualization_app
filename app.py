"""
Flask web app for the scissor-linkage optimizer.
Provides a browser form for inputs, runs the sweep, and displays results.
"""

import json
import os
import threading
import uuid
from argparse import Namespace

from flask import Flask, render_template_string, jsonify, request, send_file

from optimize import (
    sweep_offsets_and_thicknesses,
    make_interactive_plot,
)
from truss_bridge import run_fea_on_keypoints, get_material_presets, _NpEncoder

app = Flask(__name__)

# In-memory job store: job_id -> {status, progress, total, feasible, error}
jobs = {}


def _restore_jobs():
    """Restore completed jobs from result files on disk."""
    import glob
    for html_path in glob.glob(os.path.join(app.root_path, 'results_*.html')):
        basename = os.path.basename(html_path)
        job_id = basename.replace('results_', '').replace('.html', '')
        json_path = html_path.replace('.html', '.json')
        if os.path.exists(json_path) and job_id not in jobs:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                jobs[job_id] = {
                    'status': 'done',
                    'progress': len(data),
                    'total': len(data),
                    'feasible': len(data),
                    'error': None,
                    'html_path': html_path,
                    'json_path': json_path,
                }
            except Exception:
                pass


_restore_jobs()

FORM_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scissor Linkage Optimizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #090b10; color: #ccc;
      display: flex; justify-content: center; padding: 40px 20px;
    }
    .container { max-width: 700px; width: 100%; }
    h1 { color: #e0e0e0; font-size: 22px; margin-bottom: 8px; }
    .subtitle { color: #666; font-size: 12px; margin-bottom: 32px; }
    fieldset {
      border: 1px solid #1f222b; border-radius: 8px;
      padding: 20px; margin-bottom: 20px; background: #0d0f16;
    }
    legend { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; padding: 0 8px; }
    .field { display: flex; align-items: center; margin-bottom: 12px; }
    .field:last-child { margin-bottom: 0; }
    label { width: 180px; font-size: 13px; color: #aaa; flex-shrink: 0; }
    input[type="number"], input[type="text"] {
      flex: 1; background: #12141c; border: 1px solid #1f222b; border-radius: 4px;
      color: #e0e0e0; padding: 8px 12px; font-size: 13px; font-family: inherit;
    }
    input:focus { outline: none; border-color: #3a7bd5; }
    .hint { color: #555; font-size: 11px; margin-left: 8px; flex-shrink: 0; }
    .btn-row { display: flex; gap: 10px; margin-top: 8px; }
    .btn-row button { flex: 1; }
    button {
      padding: 14px; background: #3a7bd5; color: #fff;
      border: none; border-radius: 6px; font-size: 15px; font-family: inherit;
      cursor: pointer;
    }
    button:hover { background: #4a8be5; }
    button:disabled { background: #2a3a5a; cursor: not-allowed; }
    button.secondary {
      background: #1a1d27; color: #aaa; border: 1px solid #1f222b;
    }
    button.secondary:hover { background: #22262f; color: #ccc; }
    #progress {
      margin-top: 20px; padding: 16px; background: #0d0f16;
      border: 1px solid #1f222b; border-radius: 8px; display: none;
    }
    .bar-outer {
      width: 100%; height: 6px; background: #1f222b; border-radius: 3px; margin: 10px 0;
    }
    .bar-inner {
      height: 100%; background: #3a7bd5; border-radius: 3px; width: 0%;
      transition: width 0.3s;
    }
    #status { font-size: 13px; color: #aaa; }
    #error { color: #e55; margin-top: 12px; display: none; }
    .io-row { display: flex; gap: 10px; margin-bottom: 20px; }
    .io-row button { flex: 1; font-size: 13px; padding: 10px; }
    #file-input { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Scissor Linkage Optimizer</h1>
    <p class="subtitle">Enter parameters and run the sweep to find feasible solutions.</p>

    <div class="io-row">
      <button type="button" class="secondary" onclick="exportConstraints()">Export Constraints (JSON)</button>
      <button type="button" class="secondary" onclick="document.getElementById('file-input').click()">Import Constraints (JSON)</button>
      <input type="file" id="file-input" accept=".json,application/json" onchange="importConstraints(event)">
    </div>

    <form id="form">
      <fieldset>
        <legend>Geometry</legend>
        <div class="field">
          <label for="stow_width">Stow width (m)</label>
          <input type="number" id="stow_width" name="stow_width" step="any" value="5.0">
        </div>
        <div class="field">
          <label for="stow_depth">Stow depth (m)</label>
          <input type="number" id="stow_depth" name="stow_depth" step="any" placeholder="auto">
          <span class="hint">optional</span>
        </div>
        <div class="field">
          <label for="expanded_depth">Expanded depth (m)</label>
          <input type="number" id="expanded_depth" name="expanded_depth" step="any" value="70.0">
        </div>
        <div class="field">
          <label for="hinge_par">Hinge parameter</label>
          <input type="number" id="hinge_par" name="hinge_par" step="any" placeholder="auto">
          <span class="hint">optional</span>
        </div>
        <div class="field">
          <label for="n">Cells (n)</label>
          <input type="number" id="n" name="n" step="1" placeholder="auto">
          <span class="hint">optional</span>
        </div>
      </fieldset>

      <fieldset>
        <legend>Link lengths</legend>
        <div class="field">
          <label for="short">Short link (m)</label>
          <input type="number" id="short" name="short" step="any" placeholder="auto">
          <span class="hint">optional</span>
        </div>
        <div class="field">
          <label for="long">Long link (m)</label>
          <input type="number" id="long" name="long" step="any" placeholder="auto">
          <span class="hint">optional</span>
        </div>
      </fieldset>

      <fieldset>
        <legend>Offset sweep</legend>
        <div class="field">
          <label for="offset_min">Offset min</label>
          <input type="number" id="offset_min" name="offset_min" step="any" value="0.1">
        </div>
        <div class="field">
          <label for="offset_max">Offset max</label>
          <input type="number" id="offset_max" name="offset_max" step="any" value="1.0">
        </div>
        <div class="field">
          <label for="offset_steps">Offset steps</label>
          <input type="number" id="offset_steps" name="offset_steps" step="1" value="50">
        </div>
      </fieldset>

      <fieldset>
        <legend>Thickness sweep</legend>
        <div class="field">
          <label for="thickness_list">Thickness list</label>
          <input type="text" id="thickness_list" name="thickness_list" placeholder="e.g. 0.002,0.005,0.01">
          <span class="hint">or use range below</span>
        </div>
        <div class="field">
          <label for="thickness_min">Thickness min</label>
          <input type="number" id="thickness_min" name="thickness_min" step="any" value="0.01">
        </div>
        <div class="field">
          <label for="thickness_max">Thickness max</label>
          <input type="number" id="thickness_max" name="thickness_max" step="any" value="0.1">
        </div>
        <div class="field">
          <label for="thickness_steps">Thickness steps</label>
          <input type="number" id="thickness_steps" name="thickness_steps" step="1" value="8">
        </div>
      </fieldset>

      <div class="btn-row">
        <button type="submit" id="run-btn">Run Optimization</button>
      </div>
    </form>

    <div id="progress">
      <div id="status">Starting...</div>
      <div class="bar-outer"><div class="bar-inner" id="bar"></div></div>
      <div id="error"></div>
    </div>
  </div>

  <script>
    const FORM_FIELDS = [
      'stow_width', 'stow_depth', 'expanded_depth', 'hinge_par', 'n',
      'short', 'long',
      'offset_min', 'offset_max', 'offset_steps',
      'thickness_list', 'thickness_min', 'thickness_max', 'thickness_steps'
    ];

    function getFormData() {
      const data = {};
      for (const key of FORM_FIELDS) {
        const el = document.getElementById(key);
        if (el && el.value !== '') {
          data[key] = el.type === 'number' ? parseFloat(el.value) : el.value;
        }
      }
      return data;
    }

    function setFormData(data) {
      for (const key of FORM_FIELDS) {
        const el = document.getElementById(key);
        if (el && data[key] !== undefined && data[key] !== null) {
          el.value = data[key];
        } else if (el) {
          el.value = '';
        }
      }
    }

    function exportConstraints() {
      const data = getFormData();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'scissor_constraints.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    function importConstraints(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        try {
          const data = JSON.parse(e.target.result);
          setFormData(data);
        } catch (err) {
          alert('Invalid JSON file: ' + err.message);
        }
      };
      reader.readAsText(file);
      event.target.value = '';
    }

    const form = document.getElementById('form');
    const btn = document.getElementById('run-btn');
    const progressDiv = document.getElementById('progress');
    const statusEl = document.getElementById('status');
    const barEl = document.getElementById('bar');
    const errorEl = document.getElementById('error');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      btn.disabled = true;
      progressDiv.style.display = 'block';
      errorEl.style.display = 'none';
      statusEl.textContent = 'Submitting...';
      barEl.style.width = '0%';

      const data = getFormData();

      try {
        const res = await fetch('/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        const { job_id } = await res.json();
        pollProgress(job_id);
      } catch (err) {
        errorEl.textContent = 'Failed to start: ' + err.message;
        errorEl.style.display = 'block';
        btn.disabled = false;
      }
    });

    function pollProgress(jobId) {
      const iv = setInterval(async () => {
        try {
          const res = await fetch('/status/' + jobId);
          const d = await res.json();
          if (d.status === 'running') {
            const pct = d.total > 0 ? Math.round(100 * d.progress / d.total) : 0;
            barEl.style.width = pct + '%';
            statusEl.textContent = `Running... ${d.progress}/${d.total} (${d.feasible} feasible)`;
          } else if (d.status === 'done') {
            clearInterval(iv);
            barEl.style.width = '100%';
            statusEl.textContent = `Done - ${d.feasible} feasible solutions found. Redirecting...`;
            setTimeout(() => { window.location.href = '/results/' + jobId; }, 500);
          } else if (d.status === 'error') {
            clearInterval(iv);
            errorEl.textContent = d.error;
            errorEl.style.display = 'block';
            statusEl.textContent = 'Failed.';
            btn.disabled = false;
          }
        } catch (err) {
          clearInterval(iv);
          errorEl.textContent = 'Poll error: ' + err.message;
          errorEl.style.display = 'block';
          btn.disabled = false;
        }
      }, 500);
    }
  </script>
</body>
</html>
"""


def _parse_float_or_none(val):
    if val is None or val == '':
        return None
    return float(val)


def _parse_int_or_none(val):
    if val is None or val == '':
        return None
    return int(val)


@app.route('/')
def index():
    return render_template_string(FORM_HTML)


@app.route('/run', methods=['POST'])
def run():
    data = request.json
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {'status': 'running', 'progress': 0, 'total': 0, 'feasible': 0, 'error': None}

    args = Namespace(
        stow_width=_parse_float_or_none(data.get('stow_width')),
        stow_depth=_parse_float_or_none(data.get('stow_depth')),
        expanded_depth=_parse_float_or_none(data.get('expanded_depth')),
        hinge_par=_parse_float_or_none(data.get('hinge_par')),
        states=2,
        n=_parse_int_or_none(data.get('n')),
        short=_parse_float_or_none(data.get('short')),
        long=_parse_float_or_none(data.get('long')),
        offset_min=float(data.get('offset_min', 0.1)),
        offset_max=float(data.get('offset_max', 1.0)),
        offset_steps=int(data.get('offset_steps', 50)),
        thickness_list=data.get('thickness_list') or None,
        thickness_min=float(data.get('thickness_min', 0.01)),
        thickness_max=float(data.get('thickness_max', 0.1)),
        thickness_steps=int(data.get('thickness_steps', 8)),
        tee=False,
    )

    def run_job():
        try:
            def on_progress(done, total, feasible):
                jobs[job_id]['progress'] = done
                jobs[job_id]['total'] = total
                jobs[job_id]['feasible'] = feasible

            results = sweep_offsets_and_thicknesses(args, progress_callback=on_progress)

            if not results:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = 'No feasible solutions found in the sweep range.'
                return

            out_html = os.path.join(app.root_path, f'results_{job_id}.html')
            out_json = os.path.join(app.root_path, f'results_{job_id}.json')

            with open(out_json, 'w') as f:
                json.dump(results, f, indent=2)

            make_interactive_plot(results, out_html=out_html)

            jobs[job_id]['status'] = 'done'
            jobs[job_id]['feasible'] = len(results)
            jobs[job_id]['html_path'] = out_html
            jobs[job_id]['json_path'] = out_json
        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'error', 'error': 'Unknown job'}), 404
    return jsonify(job)


@app.route('/results/<job_id>')
def results(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return 'Results not ready', 404
    return send_file(job['html_path'])


@app.route('/materials')
def materials():
    return jsonify(get_material_presets())


@app.route('/run-fea', methods=['POST'])
def run_fea():
    """Run beam FEA on the current scissor geometry.

    Expects JSON body with:
      - keypoints: list of [x1,y1,z1, x2,y2,z2, w, h, layer]
      - material: str (e.g. 'aluminum_6061')
      - profile_type: str ('square', 'circle', 'rectangle')
      - hollow: bool
      - outer_dim_mm: float
      - wall_mm: float
      - P_load: float (N)
      - load_dir: str ('X', 'Y', 'Z')
    """
    data = request.json
    keypoints = data.get('keypoints')
    if not keypoints or len(keypoints) < 2:
        return jsonify({'error': 'Need at least 2 keypoints'}), 400

    try:
        result = run_fea_on_keypoints(
            keypoints,
            material=data.get('material', 'aluminum_6061'),
            profile_type=data.get('profile_type', 'square'),
            hollow=data.get('hollow', True),
            outer_dim_mm=float(data.get('outer_dim_mm', 10)),
            wall_mm=float(data.get('wall_mm', 1)),
            P_load=float(data.get('P_load', 100)),
            load_dir=data.get('load_dir', 'Y'),
            bc=data.get('bc', {}),
        )
        return app.response_class(
            response=json.dumps(result, cls=_NpEncoder),
            status=200,
            mimetype='application/json',
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
