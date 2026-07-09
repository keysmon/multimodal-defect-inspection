// src/DefectLens.js
import React, { useRef, useState } from "react";
import axios from "axios";
import "./DefectLens.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Cold-start retry: the live demo scales to zero, so the first analyze after an
// idle period commonly fails while the model warms. We retry once after a short
// backoff before surfacing an error.
const RETRY_DELAY_MS = 3000;
const RETRY_STATUS = "Model warming up - retrying...";
const ANALYZE_ERROR = "Analysis failed — is the API running?";
const COLD_START_HINT =
  "The demo scales to zero when idle - the first analysis can take a minute. Please try again.";

// GPU async path (fine-tuned VLM on a scale-to-zero SageMaker endpoint): submit
// once, then poll /vlm-status until the S3 result lands. The endpoint sleeps at
// zero instances, so the FIRST run pays a ~5 min cold start while it wakes.
const VLM_POLL_MS = 10000; // poll every 10s
const VLM_MAX_POLLS = 42; // ~7 min ceiling before giving up
const VLM_WARMING =
  "Fine-tuned model warming up on GPU - the first run can take ~5 minutes...";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A cold first request typically fails as an API Gateway 504 (integration
// timeout past the 29s cap), a 503 from a warming instance, or a dropped
// connection (ERR_NETWORK / ECONNABORTED). Those warrant one automatic retry;
// a plain application error (e.g. a 4xx/5xx bug) does not, so it is NOT retried.
function isColdStartError(err) {
  const status = err?.response?.status;
  if (status === 503 || status === 504) return true;
  return err?.code === "ERR_NETWORK" || err?.code === "ECONNABORTED";
}

// One-click example gallery. Assets live in public/gallery/ (built by
// scripts/build_gallery_assets.py from CC BY datasets; see that folder's
// ATTRIBUTION.md). Each entry loads its image + inspector note and runs analyze.
const GALLERY_EXAMPLES = [
  {
    image: "sdnet-wall-crack.jpg",
    caption: "Concrete wall - crack",
    note: "Diagonal hairline crack on an exterior concrete wall; checking whether it is active.",
  },
  {
    image: "sdnet-pavement-crack.jpg",
    caption: "Pavement - crack",
    note: "Transverse crack across a concrete slab near an expansion joint.",
  },
  {
    image: "metu-crack.jpg",
    caption: "Facade - crack",
    note: "Vertical crack on a campus building facade; width not yet measured.",
  },
  {
    image: "sdnet-wall-clean.jpg",
    caption: "Concrete wall - no defect",
    note: "Baseline concrete wall section with no visible cracking.",
  },
  {
    image: "sdnet-deck-clean.jpg",
    caption: "Bridge deck - no defect",
    note: "Concrete bridge deck, routine condition check.",
  },
  {
    image: "metu-clean.jpg",
    caption: "Facade - no defect",
    note: "Clean facade panel used as a reference image.",
  },
];

// Severity band -> display styling (spec: structural/urgent/monitor/cosmetic).
const SEVERITY_STYLES = {
  structural: { background: "#c0392b", color: "#fff", label: "Structural" },
  urgent: { background: "#e67e22", color: "#fff", label: "Urgent" },
  monitor: { background: "#f1c40f", color: "#222", label: "Monitor" },
  cosmetic: { background: "#27ae60", color: "#fff", label: "Cosmetic" },
};

function severityStyle(severity) {
  return (
    SEVERITY_STYLES[severity] || {
      background: "#95a5a6",
      color: "#fff",
      label: severity || "Unknown",
    }
  );
}

// Shared card list renderer for both /analyze and /search results.
function CardList({ cards }) {
  if (!cards || cards.length === 0) {
    return null;
  }
  return (
    <ul className="card-list">
      {cards.map((card) => (
        <li key={card.id} className="guidance-card">
          <div className="card-header">
            <h3 className="card-title">{card.title}</h3>
            <span
              className="card-severity-tag"
              style={{
                backgroundColor: severityStyle(card.severity).background,
                color: severityStyle(card.severity).color,
              }}
            >
              {card.severity}
            </span>
          </div>
          <p className="card-passage">{card.passage}</p>
          <p className="card-citation">{card.citation}</p>
          <a
            href={card.source_url}
            target="_blank"
            rel="noreferrer"
            className="card-source-link"
          >
            {card.source_name}
          </a>
        </li>
      ))}
    </ul>
  );
}

function buildReportMarkdown(analyzeResult) {
  const date = new Date().toISOString().slice(0, 10);
  const lines = [];

  lines.push("# DefectLens Report");
  lines.push("");
  lines.push(`- Date: ${date}`);
  lines.push(`- Filename: ${analyzeResult.filename}`);
  lines.push(`- Severity: ${severityStyle(analyzeResult.severity).label}`);
  if (analyzeResult.audio) {
    lines.push(
      `- Combined severity: ${severityStyle(analyzeResult.combined_severity).label}`
    );
  }
  if (analyzeResult.note) lines.push(`- Inspector note: ${analyzeResult.note}`);
  lines.push("");

  lines.push("## Ranked classes");
  analyzeResult.classes.forEach((c, i) => {
    lines.push(`${i + 1}. ${c.label} (score: ${Number(c.score).toFixed(3)})`);
  });
  lines.push("");

  if (analyzeResult.description) {
    lines.push("## Description");
    lines.push(analyzeResult.description);
    lines.push("");
  }

  lines.push("## Guidance cards");
  analyzeResult.cards.forEach((card) => {
    lines.push(`### ${card.title}`);
    lines.push(`- Severity: ${card.severity}`);
    lines.push(`- Passage: ${card.passage}`);
    lines.push(`- Citation: ${card.citation}`);
    lines.push(`- Source: ${card.source_url}`);
    lines.push("");
  });

  if (analyzeResult.audio) {
    lines.push("## Equipment audio");
    lines.push(`- Band: ${analyzeResult.audio.band}`);
    lines.push(`- Score: ${Number(analyzeResult.audio.score).toFixed(3)}`);
    lines.push(`- Severity: ${analyzeResult.audio.severity}`);
    if (analyzeResult.audio.cards && analyzeResult.audio.cards.length) {
      lines.push("- Guidance:");
      analyzeResult.audio.cards.forEach((card) => {
        lines.push(`  - ${card.title}`);
      });
    }
    lines.push("");
  }

  return lines.join("\n");
}

function DefectLens() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [note, setNote] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeStatus, setAnalyzeStatus] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);

  // GPU (fine-tuned VLM) re-run of the just-analyzed image.
  const [isVlmRunning, setIsVlmRunning] = useState(false);
  const [vlmStatus, setVlmStatus] = useState("");
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmError, setVlmError] = useState("");

  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);

  const [error, setError] = useState("");

  // The audio <input> is uncontrolled: clearing selectedAudio state on image
  // change does NOT reset the DOM value, so re-picking the same wav fires no
  // change event and the file is silently dropped. Reset the element too.
  const audioInputRef = useRef(null);

  const resetVlm = () => {
    setIsVlmRunning(false);
    setVlmStatus("");
    setVlmResult(null);
    setVlmError("");
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file || null);
    setImagePreview(file ? URL.createObjectURL(file) : null);
    setNote("");
    setSelectedAudio(null);
    if (audioInputRef.current) audioInputRef.current.value = "";
    setAnalyzeResult(null);
    resetVlm();
    setError("");
  };

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    setSelectedAudio(file || null);
  };

  const handleAnalyze = async (overrides = {}) => {
    // Overrides let the gallery run analyze with a freshly-fetched file/note
    // synchronously, without waiting for the async state updates to flush.
    const file = overrides.file ?? selectedFile;
    const noteText = overrides.note ?? note;
    const audioFile = "audio" in overrides ? overrides.audio : selectedAudio;
    if (!file) {
      setError("Please select an image first.");
      return;
    }
    setIsAnalyzing(true);
    setError("");
    setAnalyzeStatus("");
    resetVlm(); // a fresh analysis invalidates any prior GPU re-run

    const formData = new FormData();
    formData.append("file", file);
    if (noteText.trim()) formData.append("note", noteText.trim());
    if (audioFile) formData.append("audio", audioFile);

    const postAnalyze = () =>
      axios.post(`${API}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

    let retried = false;
    try {
      let response;
      try {
        response = await postAnalyze();
      } catch (err) {
        if (!isColdStartError(err)) throw err;
        retried = true;
        setAnalyzeStatus(RETRY_STATUS);
        await sleep(RETRY_DELAY_MS);
        response = await postAnalyze();
      }
      setAnalyzeResult({ ...response.data, filename: file.name });
    } catch (err) {
      console.error("Error during analyze:", err);
      setError(retried ? `${ANALYZE_ERROR} ${COLD_START_HINT}` : ANALYZE_ERROR);
      setAnalyzeResult(null);
    } finally {
      setIsAnalyzing(false);
      setAnalyzeStatus("");
    }
  };

  const handleRunGpu = async () => {
    // Re-run the just-analyzed image through the fine-tuned VLM on the GPU async
    // endpoint. Submit once, then poll /vlm-status until the S3 result lands.
    if (!selectedFile || isVlmRunning) return;
    setIsVlmRunning(true);
    setVlmError("");
    setVlmResult(null);
    setVlmStatus(VLM_WARMING);

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (note.trim()) formData.append("note", note.trim());

    try {
      const submit = await axios.post(`${API}/analyze-vlm`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const { output_location, failure_location } = submit.data;

      let settled = false;
      for (let i = 0; i < VLM_MAX_POLLS && !settled; i++) {
        const params = { output_location };
        if (failure_location) params.failure_location = failure_location;
        // axios treats 202 (pending) as success, so only ready/failed bodies
        // resolve the loop; anything else means "still warming, keep polling".
        const poll = await axios.get(`${API}/vlm-status`, { params });
        if (poll.data.status === "ready") {
          setVlmResult({ classes: poll.data.classes });
          settled = true;
        } else if (poll.data.status === "failed") {
          setVlmError("The fine-tuned model run failed. Please try again.");
          settled = true;
        } else if (i < VLM_MAX_POLLS - 1) {
          await sleep(VLM_POLL_MS);
        }
      }
      if (!settled) {
        setVlmError(
          "The fine-tuned model is taking longer than expected. Please try again."
        );
      }
    } catch (err) {
      console.error("Error during GPU analyze:", err);
      if (err?.response?.status === 503) {
        setVlmError("The fine-tuned GPU model isn't deployed for this demo.");
      } else {
        setVlmError("Fine-tuned model request failed. Please try again.");
      }
    } finally {
      setIsVlmRunning(false);
      setVlmStatus("");
    }
  };

  const handleGalleryExample = async (example) => {
    setError("");
    let file;
    try {
      const response = await fetch(
        `${process.env.PUBLIC_URL}/gallery/${example.image}`
      );
      const blob = await response.blob();
      file = new File([blob], example.image, {
        type: blob.type || "image/jpeg",
      });
    } catch (err) {
      console.error("Error loading gallery example:", err);
      setError("Couldn't load the example image. Please try uploading one.");
      return;
    }
    setSelectedFile(file);
    setImagePreview(URL.createObjectURL(file));
    setNote(example.note);
    setSelectedAudio(null);
    if (audioInputRef.current) audioInputRef.current.value = "";
    setAnalyzeResult(null);
    await handleAnalyze({ file, note: example.note, audio: null });
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError("Please enter a search query.");
      return;
    }
    setIsSearching(true);
    setError("");

    try {
      const response = await axios.post(`${API}/search`, {
        query: searchQuery,
      });
      setSearchResult({ cards: response.data.cards });
    } catch (err) {
      console.error("Error during search:", err);
      setError("Search failed — is the API running?");
      setSearchResult(null);
    } finally {
      setIsSearching(false);
    }
  };

  const handleExport = () => {
    if (!analyzeResult) {
      return;
    }
    const markdown = buildReportMarkdown(analyzeResult);
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const date = new Date().toISOString().slice(0, 10);
    const link = document.createElement("a");
    link.href = url;
    link.download = `defectlens-report-${date}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const topClasses = analyzeResult ? analyzeResult.classes.slice(0, 3) : [];
  const hasAudio = Boolean(analyzeResult && analyzeResult.audio);
  const bannerSeverity = analyzeResult
    ? hasAudio
      ? analyzeResult.combined_severity
      : analyzeResult.severity
    : null;
  const bandStyle = bannerSeverity ? severityStyle(bannerSeverity) : null;

  return (
    <div className="defectlens-container">
      <header className="dl-header">
        <h1 className="dl-title">
          DefectLens — building-defect inspection assistant
        </h1>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="gallery-section">
        <h2 className="gallery-title">Try an example</h2>
        <p className="gallery-subtitle">
          One click loads a sample photo and inspector note, then runs the
          analysis.
        </p>
        <div className="gallery-grid">
          {GALLERY_EXAMPLES.map((example) => (
            <button
              key={example.image}
              type="button"
              className="gallery-tile"
              onClick={() => handleGalleryExample(example)}
              disabled={isAnalyzing}
              aria-label={`Load example: ${example.caption}`}
            >
              <img
                src={`${process.env.PUBLIC_URL}/gallery/${example.image}`}
                alt={example.caption}
                className="gallery-thumb"
                loading="lazy"
              />
              <span className="gallery-caption">{example.caption}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
          data-testid="file-input"
          aria-label="Upload image"
        />
        {imagePreview && (
          <img
            src={imagePreview}
            alt="Selected preview"
            className="preview-image"
          />
        )}
        <textarea
          className="note-input"
          placeholder="Optional inspector note (e.g., 'musty smell, below upstairs bathroom')"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          maxLength={500}
        />
        <input
          type="file"
          accept=".wav,audio/wav"
          ref={audioInputRef}
          onChange={handleAudioChange}
          className="audio-input"
          data-testid="audio-input"
          aria-label="Upload equipment audio (optional)"
        />
        {selectedAudio && (
          <span className="audio-filename">{selectedAudio.name}</span>
        )}
        <button
          onClick={() => handleAnalyze()}
          disabled={isAnalyzing}
          className="analyze-button"
        >
          {isAnalyzing ? "Analyzing..." : "Analyze"}
        </button>
        {analyzeStatus && <p className="analyze-status">{analyzeStatus}</p>}
      </section>

      {analyzeResult && (
        <section className="results-section">
          <div
            className="severity-banner"
            style={{
              backgroundColor: bandStyle.background,
              color: bandStyle.color,
            }}
          >
            {hasAudio ? "Combined severity" : "Severity"}: {bandStyle.label}
          </div>

          <div className="rank-chips">
            {topClasses.map((c, i) => (
              <span key={c.label} className="rank-chip">
                {`${i + 1}. ${c.label.replace(/_/g, " ")}`}
                {typeof c.score === "number" && (
                  <span className="rank-score">
                    {`${Math.round(c.score * 100)}%`}
                  </span>
                )}
              </span>
            ))}
            {analyzeResult.classifier && (
              <span
                className="classifier-badge"
                title={
                  analyzeResult.classifier === "vlm-qlora"
                    ? "Classified by the fine-tuned Qwen2.5-VL model (macro top-1 0.851 on the frozen test split)"
                    : "Classified by the CLIP retrieval-fusion baseline"
                }
              >
                {analyzeResult.classifier === "vlm-qlora"
                  ? "fine-tuned VLM"
                  : "CLIP baseline"}
              </span>
            )}
          </div>

          {analyzeResult.description && (
            <p className="description">{analyzeResult.description}</p>
          )}

          <CardList cards={analyzeResult.cards} />

          {hasAudio && (
            <div className="audio-panel">
              <h2 className="audio-panel-title">Equipment audio</h2>
              <div className="audio-summary">
                <span
                  className="audio-band-chip"
                  style={{
                    backgroundColor: severityStyle(analyzeResult.audio.severity)
                      .background,
                    color: severityStyle(analyzeResult.audio.severity).color,
                  }}
                >
                  {analyzeResult.audio.band.replace(/_/g, " ")}
                </span>
                <span className="audio-score">
                  score: {Number(analyzeResult.audio.score).toFixed(3)}
                </span>
              </div>
              <CardList cards={analyzeResult.audio.cards} />
            </div>
          )}

          <div className="gpu-panel">
            <button
              onClick={handleRunGpu}
              disabled={isVlmRunning}
              className="gpu-button"
            >
              {isVlmRunning
                ? "Running fine-tuned model..."
                : "Run fine-tuned model (GPU, ~5 min cold)"}
            </button>
            {vlmStatus && <p className="analyze-status">{vlmStatus}</p>}
            {vlmError && <p className="vlm-error">{vlmError}</p>}
            {vlmResult && (
              <div className="rank-chips vlm-chips">
                {vlmResult.classes.slice(0, 3).map((c, i) => (
                  <span key={c.label} className="rank-chip">
                    {`${i + 1}. ${c.label.replace(/_/g, " ")}`}
                    {typeof c.score === "number" && (
                      <span className="rank-score">
                        {`${Math.round(c.score * 100)}%`}
                      </span>
                    )}
                  </span>
                ))}
                <span
                  className="classifier-badge"
                  title="Classified by the fine-tuned Qwen2.5-VL model on the GPU async endpoint (macro top-1 0.851 on the frozen test split)"
                >
                  fine-tuned VLM (GPU)
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      <section className="export-section">
        <button
          onClick={handleExport}
          disabled={!analyzeResult}
          className="export-button"
        >
          Export report (markdown)
        </button>
      </section>

      <section className="search-section">
        <h2>Search guidance</h2>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search defect guidance..."
          className="search-input"
        />
        <button
          onClick={handleSearch}
          disabled={isSearching}
          className="search-button"
        >
          {isSearching ? "Searching..." : "Search"}
        </button>

        {searchResult && <CardList cards={searchResult.cards} />}
      </section>
    </div>
  );
}

export default DefectLens;
