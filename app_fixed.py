import os
import re
import subprocess
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0,
)


def clean_response(response):
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_code(text):
    match = re.search(r"```(?:verilog|systemverilog|v)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def generate_verilog(task):
    prompt = f"""
You are an expert Verilog HDL designer.

USER TASK:
{task}

Generate the complete synthesizable Verilog design.
Use standard Verilog-2001/Verilog-2005 only. Do not use SystemVerilog.
Match the task exactly. Return ONLY the Verilog source code in one code block.
Check module name, ports, widths, clock/reset behavior, assignments and syntax.
"""
    return extract_code(clean_response(llm.invoke(prompt)))


def generate_testbench(task, design):
    prompt = f"""
You are an expert Verilog verification engineer.

USER TASK:
{task}

VERILOG DESIGN:
{design}

Generate a complete matching Verilog-2005 testbench.
Use the EXACT module name, port names and widths from the design.
Instantiate the design correctly. For clocked circuits generate a clock.
For reset circuits test reset behavior. Apply meaningful test cases.
Use $display and $finish. Return ONLY testbench code in one code block.
Do not use SystemVerilog.
"""
    return extract_code(clean_response(llm.invoke(prompt)))


def simulate(design, testbench):
    d = tempfile.mkdtemp()
    design_file = os.path.join(d, "design.v")
    tb_file = os.path.join(d, "testbench.v")
    out_file = os.path.join(d, "simulation.out")

    with open(design_file, "w", encoding="utf-8") as f:
        f.write(design)
    with open(tb_file, "w", encoding="utf-8") as f:
        f.write(testbench)

    try:
        compile_result = subprocess.run(
            ["iverilog", "-g2005", "-o", out_file, design_file, tb_file],
            capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        return {
            "status": "ERROR",
            "output": "",
            "error": "Icarus Verilog is not installed on the Render service. Install iverilog or deploy with Docker."
        }

    if compile_result.returncode != 0:
        return {"status": "COMPILE ERROR", "output": compile_result.stdout, "error": compile_result.stderr}

    try:
        run_result = subprocess.run(
            ["vvp", out_file], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        return {"status": "ERROR", "output": "", "error": "vvp was not found. Icarus installation is incomplete."}

    if run_result.returncode != 0:
        return {"status": "SIMULATION ERROR", "output": run_result.stdout, "error": run_result.stderr}

    return {"status": "PASS", "output": run_result.stdout, "error": run_result.stderr}


def verify(task, design, testbench, sim):
    prompt = f"""
You are a senior Verilog verification engineer.
USER TASK: {task}
VERILOG: {design}
TESTBENCH: {testbench}
ACTUAL ICARUS RESULT: STATUS={sim['status']} OUTPUT={sim['output']} ERROR={sim['error']}

Return exactly:
VERIFICATION: PASS or FAIL
REASON: <short reason>
SIMULATION_RESULT: <short explanation>
Do not invent simulation results.
"""
    return clean_response(llm.invoke(prompt))


def repair(task, design, testbench, sim):
    prompt = f"""
Fix the actual Verilog/Icarus problem while preserving the user's task.
Use Verilog-2005 only.

USER TASK:
{task}
CURRENT VERILOG:
{design}
CURRENT TESTBENCH:
{testbench}
ACTUAL STATUS: {sim['status']}
OUTPUT: {sim['output']}
ERROR: {sim['error']}

Return exactly:
VERILOG_CODE:
<complete corrected Verilog>
TESTBENCH_CODE:
<complete corrected testbench>
No explanations.
"""
    text = clean_response(llm.invoke(prompt))
    vm = re.search(r"VERILOG_CODE:\s*(.*?)(?=TESTBENCH_CODE:)", text, re.DOTALL | re.IGNORECASE)
    tm = re.search(r"TESTBENCH_CODE:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not vm or not tm:
        raise ValueError("Repair agent did not return both files.")
    return extract_code(vm.group(1)), extract_code(tm.group(1))


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verilog Code Generation Agent</title>
<style>
body{margin:0;padding:30px;background:#f4f5f7;font-family:Arial,sans-serif}.box{max-width:1350px;margin:auto;background:#fff;padding:38px;border-radius:15px;box-sizing:border-box}h1{font-size:43px;margin:0 0 22px}.desc{font-size:21px;margin-bottom:22px}textarea{width:100%;min-height:205px;box-sizing:border-box;padding:16px;font:18px monospace;border:1px solid #888;border-radius:4px;resize:vertical}button{width:100%;margin-top:18px;padding:17px;border:0;border-radius:10px;background:#222;color:#fff;font-size:20px;cursor:pointer}button:disabled{opacity:.6;cursor:wait}#status{margin-top:20px;font-size:18px;font-weight:bold}#result{display:none;margin-top:20px;background:#111;color:#eee;padding:22px;border-radius:10px;white-space:pre-wrap;font:15px monospace;line-height:1.45;overflow:auto}.err{color:#b00020}</style></head>
<body><div class="box"><h1>Verilog Code Generation Agent</h1><div class="desc">Generate Verilog + matching testbench + real Icarus simulation + verification.</div>
<textarea id="task">Design a 4-bit up counter using Verilog with a clock and an active-high reset. The counter should increment by 1 on every positive edge of the clock and reset to 0 when reset is high.</textarea>
<button id="btn" type="button">Generate &amp; Simulate</button><div id="status"></div><pre id="result"></pre></div>
<script>
const btn=document.getElementById('btn'), task=document.getElementById('task'), status=document.getElementById('status'), result=document.getElementById('result');
btn.addEventListener('click', async()=>{
 const user_input=task.value.trim();
 if(!user_input){status.innerHTML='<span class="err">Please enter a Verilog task.</span>';return;}
 btn.disabled=true; btn.textContent='Generating & Simulating...'; status.textContent='Running Verilog generation, testbench generation and Icarus simulation...'; result.style.display='none'; result.textContent='';
 try{
  const r=await fetch('/agent/playground/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_input})});
  const data=await r.json();
  if(!r.ok || !data.success) throw new Error(data.error||'Server returned an error.');
  status.textContent='Completed.'; result.textContent=data.final_output||'No result returned.'; result.style.display='block';
 }catch(e){status.innerHTML='<span class="err">Error: '+String(e.message).replace(/</g,'&lt;')+'</span>';}
 finally{btn.disabled=false;btn.textContent='Generate & Simulate';}
});
</script></body></html>'''


@app.route("/")
def home():
    return '<a href="/agent/playground/">Open Verilog Agent Playground</a>'


@app.route("/agent/playground/")
def playground():
    return render_template_string(HTML)


@app.route("/agent/playground/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        task = str(data.get("user_input", "")).strip()
        if not task:
            return jsonify({"success": False, "error": "Verilog task is empty."}), 400

        print("NEW REQUEST:", task, flush=True)
        design = generate_verilog(task)
        testbench = generate_testbench(task, design)
        sim = simulate(design, testbench)

        for attempt in range(3):
            if sim["status"] == "PASS":
                break
            print(f"Repair attempt {attempt+1}/3", flush=True)
            if sim["status"] == "ERROR" and "Icarus Verilog is not installed" in sim["error"]:
                break
            design, testbench = repair(task, design, testbench, sim)
            sim = simulate(design, testbench)

        verification = verify(task, design, testbench, sim)
        final = f"""============================================================
                 VERILOG SOLUTION
============================================================

VERILOG CODE:
------------------------------------------------------------
{design}

TESTBENCH CODE:
------------------------------------------------------------
{testbench}

SIMULATION RESULT:
------------------------------------------------------------
STATUS: {sim['status']}

SIMULATION OUTPUT:
{sim['output']}

SIMULATION ERROR:
{sim['error']}

VERIFICATION:
------------------------------------------------------------
{verification}

============================================================"""
        return jsonify({"success": True, "final_output": final})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
