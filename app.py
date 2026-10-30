"""
app.py — Portfolio demo for Deep Research RAG.
Dark-mode FastAPI web interface over the 7-stage RAG pipeline.
No additional dependencies beyond what is already in requirements.txt.
Run: python app.py
Then open: http://localhost:7860
"""

import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from src.observability.logging_config import new_request_id, setup_logging
from src.observability.tracing import log_tracing_status
from src.pipeline.rag_pipeline import load_resources, run_pipeline

load_dotenv()      # first, so LOG_LEVEL and the LangSmith variables are visible
setup_logging()    # configure logging once, before anything logs

logger = logging.getLogger(__name__)

log_tracing_status(logger)
logger.info("Loading pipeline resources...")
_resources = load_resources()
logger.info("Ready. Open http://localhost:%s", os.getenv("PORT", "7860"))

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest, response: Response):
    # One ID per request: stamped on every log line for this request,
    # attached to the LangSmith trace, and returned to the browser.
    request_id = new_request_id()
    response.headers["X-Request-ID"] = request_id
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a question.")
    logger.info("Demo UI request: %d-char question", len(question))
    try:
        result = run_pipeline(
            question,
            _resources,
            langsmith_extra={
                "metadata": {"request_id": request_id, "entrypoint": "demo-ui"},
                "tags": ["demo-ui"],
            },
        )
    except Exception:
        # Full error goes to the server log; the browser gets a short, safe message
        # with a real error status instead of a 200 that looks like an answer.
        logger.exception("Pipeline failed for question: %r", question)
        raise HTTPException(
            status_code=500,
            detail=f"The pipeline could not answer this question. Please try again later. (request id {request_id})",
        )
    # Return only what the UI renders. Internal fields (retrieved contexts,
    # HyDE passage, pool sizes) stay on the server.
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "model": result.get("model", ""),
        "request_id": request_id,
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Research RAG</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0b0b12; font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; color: #e4e4f0; min-height: 100vh; }
  .container { max-width: 1180px; margin: 0 auto; padding: 0 24px 60px; }
  .header { padding: 36px 0 24px; border-bottom: 1px solid #1c1c30; margin-bottom: 32px; }
  .title { font-size: 26px; font-weight: 700; color: #eeeef8; letter-spacing: -0.5px; margin-bottom: 12px; }
  .badge { display: inline-block; background: #12122a; border: 1px solid #2a2a50; border-radius: 20px; padding: 2px 10px; font-size: 11px; color: #5b7cf7; margin-right: 6px; margin-bottom: 4px; font-weight: 600; letter-spacing: 0.3px; }
  .desc { font-size: 13.5px; color: #686890; line-height: 1.6; }
  .layout { display: grid; grid-template-columns: 260px 1fr; gap: 32px; align-items: start; }
  @media (max-width: 720px) { .layout { grid-template-columns: 1fr; } }
  .sec-label { display: block; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.4px; color: #5b7cf7; margin-bottom: 12px; }
  .side-hr { border: none; border-top: 1px solid #1c1c30; margin: 18px 0; }
  .corpus-card { background: #111120; border: 1px solid #1c1c30; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; }
  .corpus-name { font-size: 12.5px; font-weight: 600; color: #d8d8f0; margin-bottom: 3px; }
  .corpus-type { font-size: 11.5px; color: #686890; margin-bottom: 3px; }
  .corpus-coverage { font-size: 11px; color: #4a4a68; }
  .pipeline-step { display: flex; align-items: center; gap: 9px; padding: 5px 0; font-size: 12px; color: #8888b0; border-bottom: 1px solid #131326; }
  .pipeline-step:last-child { border-bottom: none; }
  .step-num { display: inline-flex; align-items: center; justify-content: center; min-width: 20px; height: 20px; background: #141430; border: 1px solid #202050; border-radius: 50%; font-size: 9px; font-weight: 700; color: #5b7cf7; flex-shrink: 0; }
  .ex-btn { display: block; width: 100%; background: #111120; border: 1px solid #1c1c30; border-radius: 6px; color: #b0b0d0; font-size: 12px; text-align: left; padding: 8px 12px; margin-bottom: 5px; cursor: pointer; line-height: 1.4; transition: border-color 0.15s, background 0.15s, color 0.15s; font-family: inherit; }
  .ex-btn:hover { border-color: #3a3a70; background: #17172e; color: #e4e4f0; }
  .field-label { display: block; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #5b7cf7; margin-bottom: 8px; }
  #question { width: 100%; background: #111120; border: 1px solid #1c1c30; border-radius: 8px; color: #e4e4f0; font-size: 14px; font-family: inherit; line-height: 1.6; padding: 12px 16px; resize: vertical; min-height: 72px; transition: border-color 0.15s; outline: none; }
  #question:focus { border-color: #5b7cf7; box-shadow: 0 0 0 3px rgba(91,124,247,0.10); }
  #question::placeholder { color: #3a3a58; }
  #ask-btn { width: 100%; margin-top: 12px; background: #5b7cf7; border: none; border-radius: 8px; color: #fff; font-size: 14px; font-weight: 600; font-family: inherit; height: 44px; cursor: pointer; letter-spacing: 0.3px; transition: background 0.15s; }
  #ask-btn:hover { background: #7090ff; }
  #ask-btn:active { background: #4a6ae0; }
  #ask-btn:disabled { background: #2a2a50; color: #686890; cursor: not-allowed; }
  .output-block { margin-top: 20px; border-radius: 8px; padding: 18px 20px; border: 1px solid #1c1c30; }
  #answer-box { background: #111120; min-height: 60px; }
  #sources-box { background: #0d0d1a; margin-top: 12px; }
  .output-block .field-label { margin-bottom: 12px; }
  #answer-text { font-size: 14px; line-height: 1.78; color: #e4e4f0; white-space: pre-wrap; }
  #sources-text { font-size: 13px; line-height: 1.7; color: #9090b8; }
  .source-item { margin-bottom: 4px; }
  .source-item code { background: #18214a; border: 1px solid #283880; color: #7b96ff; border-radius: 4px; padding: 1px 6px; font-size: 12px; }
  .spinner { display: none; width: 16px; height: 16px; border: 2px solid #3a3a6a; border-top-color: #5b7cf7; border-radius: 50%; animation: spin 0.7s linear infinite; margin-left: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .footer { border-top: 1px solid #1c1c30; padding-top: 20px; margin-top: 48px; text-align: center; font-size: 12px; color: #484868; }
  .footer a { color: #5b7cf7; text-decoration: none; }
  .footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="title">Deep Research RAG</div>
    <div class="badges">
      <span class="badge">HyDE</span>
      <span class="badge">Hybrid Retrieval</span>
      <span class="badge">RRF Fusion</span>
      <span class="badge">Cross-Encoder Reranking</span>
      <span class="badge">Grounded Generation</span>
    </div>
    <div class="desc">A 7-stage RAG pipeline over dense Infosys business documents. The model answers only from retrieved context, and each answer lists the document pages it cites.</div>
  </div>
  <div class="layout">
    <aside>
      <span class="sec-label">Indexed Corpus</span>
      <div class="corpus-card">
        <div class="corpus-name">Infosys Integrated Annual Report 2025-26</div>
        <div class="corpus-type">Annual Report · 383 pages</div>
        <div class="corpus-coverage">Financials, business segments, geography, strategy, workforce</div>
      </div>
      <div class="corpus-card">
        <div class="corpus-name">Infosys AGM Transcript 2026</div>
        <div class="corpus-type">AGM Transcript · 50 pages</div>
        <div class="corpus-coverage">Management remarks and shareholder Q&amp;A</div>
      </div>
      <div class="corpus-card">
        <div class="corpus-name">Additional Information 2025-26</div>
        <div class="corpus-type">Supplementary Report · 8 pages</div>
        <div class="corpus-coverage">Multi-year financial and operating summary</div>
      </div>
      <div class="corpus-card">
        <div class="corpus-name">CSR Projects 2025-26</div>
        <div class="corpus-type">CSR Report · 4 pages</div>
        <div class="corpus-coverage">Corporate social responsibility projects</div>
      </div>
      <div class="corpus-coverage" style="margin-top:6px;">444 pages of text · 3,439 chunks</div>
      <hr class="side-hr">
      <span class="sec-label">Pipeline</span>
      <div id="pipeline-steps"></div>
      <hr class="side-hr">
      <span class="sec-label">Example Questions</span>
      <div id="examples"></div>
    </aside>
    <main>
      <label class="field-label" for="question">Question</label>
      <textarea id="question" rows="3" placeholder="Ask anything about the indexed documents…"></textarea>
      <button id="ask-btn" onclick="ask()">Ask <span class="spinner" id="spinner"></span></button>
      <div class="output-block" id="answer-box">
        <span class="field-label">Answer</span>
        <div id="answer-text" style="color:#3a3a58;">Answer will appear here.</div>
      </div>
      <div class="output-block" id="sources-box">
        <span class="field-label">Sources</span>
        <div id="sources-text" style="color:#3a3a58;">Sources will appear here.</div>
      </div>
    </main>
  </div>
  <div class="footer">
    Built by <strong style="color:#9090b8">Abhishek Vinod</strong> &nbsp;·&nbsp;
    <a href="https://github.com/CoderAbhishek/deep-research-rag" target="_blank">GitHub</a>
    &nbsp;·&nbsp; Groq · ChromaDB · sentence-transformers · FastAPI
  </div>
</div>
<script>
const PIPELINE_STEPS = ["HyDE — hypothetical answer generation","Dense retrieval — ChromaDB + MiniLM-L6","BM25 sparse retrieval","RRF fusion","Cross-encoder reranking","Page-level deduplication","Groq LLM generation"];
const EXAMPLES = ["What is Infosys Cobalt?","What was Infosys's total revenue in FY26?","What is Infosys's largest business segment by revenue?","How many employees does Infosys have globally?","What did the CEO say about AI strategy at the AGM?","What is the North America revenue for Infosys in FY26?"];
const stepsEl = document.getElementById("pipeline-steps");
PIPELINE_STEPS.forEach((s,i) => { stepsEl.innerHTML += `<div class="pipeline-step"><span class="step-num">${i+1}</span>${s}</div>`; });
const examplesEl = document.getElementById("examples");
EXAMPLES.forEach(q => { const btn = document.createElement("button"); btn.className="ex-btn"; btn.textContent=q; btn.onclick=()=>{document.getElementById("question").value=q;}; examplesEl.appendChild(btn); });
async function ask() {
  const q = document.getElementById("question").value.trim();
  if (!q) return;
  const btn = document.getElementById("ask-btn");
  const spinner = document.getElementById("spinner");
  btn.disabled = true; spinner.style.display = "inline-block";
  document.getElementById("answer-text").textContent = "Thinking…";
  document.getElementById("answer-text").style.color = "#686890";
  document.getElementById("sources-text").textContent = "";
  try {
    const res = await fetch("/query", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({question: q}) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = (typeof data.detail === "string") ? data.detail : ("Request failed with status " + res.status + ".");
      document.getElementById("answer-text").style.color = "#f08a8a";
      document.getElementById("answer-text").textContent = "Error: " + msg;
      document.getElementById("sources-text").textContent = "";
      return;
    }
    document.getElementById("answer-text").style.color = "#e4e4f0";
    document.getElementById("answer-text").textContent = data.answer || "No answer returned.";
    const srcs = data.sources || [];
    if (srcs.length > 0) {
      document.getElementById("sources-text").innerHTML = srcs.map(s => `<div class="source-item"><strong>[${s.source_num}]</strong> <code>${s.file_name}</code> — Page <strong>${s.page_number}</strong></div>`).join("");
    } else { document.getElementById("sources-text").textContent = "No sources cited."; }
    if (data.request_id) { document.getElementById("sources-text").insertAdjacentHTML("beforeend", `<div class="source-item" style="margin-top:10px;color:#4a4a68;">Request ID: <code>${data.request_id}</code></div>`); }
  } catch(err) {
    document.getElementById("answer-text").style.color = "#f08a8a";
    document.getElementById("answer-text").textContent = "Error: could not reach the server (" + err.message + ").";
  } finally { btn.disabled=false; spinner.style.display="none"; }
}
document.getElementById("question").addEventListener("keydown", e => { if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask();} });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)