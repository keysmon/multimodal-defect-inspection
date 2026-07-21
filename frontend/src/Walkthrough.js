// Walkthrough diagnostic report (P3): N photos + a visit note -> a grounded,
// cited initial-diagnostic report from the async /walkthrough-jobs path.
import React, { useRef, useState } from "react";
import axios from "axios";
import { isColdStartError, sleep } from "./apiHelpers";

export const MAX_WALKTHROUGH_PHOTOS = 10;
const DEFAULT_POLL_MS = 1500;
// ~4.5 min ceiling: a cold worker loads models (~30s) then makes two Haiku
// calls over up to 10 photos; generous but bounded.
const MAX_POLLS = 180;
const SUBMIT_RETRY_DELAY_MS = 3000;

const PRIORITY_RANK = { high: 0, medium: 1, low: 2 };
const CAP_MESSAGE = `A walkthrough is capped at ${MAX_WALKTHROUGH_PHOTOS} photos.`;
const FAILED_MESSAGE = "Walkthrough failed. Please try again.";
const TIMEOUT_MESSAGE =
  "The walkthrough is taking longer than expected. Please try again.";

export function buildWalkthroughMarkdown(report, photoNames = {}) {
  const date = new Date().toISOString().slice(0, 10);
  const cardTitle = (cid) => report.cards?.[cid]?.title || cid;
  const citeLine = (ids) =>
    ids && ids.length ? `Citations: ${ids.map(cardTitle).join("; ")}` : null;
  const lines = [];

  lines.push("# Walkthrough diagnostic report");
  lines.push("");
  lines.push(`- Date: ${date}`);
  lines.push(`- **${report.disclaimer}**`);
  lines.push("");

  lines.push("## Overall assessment");
  lines.push(report.summary.overall_assessment);
  const assessmentCites = citeLine(report.summary.assessment_citations);
  lines.push(
    assessmentCites || "(auto-summary derived from the cited findings below)"
  );
  lines.push("");

  if (report.summary.answers.length) {
    lines.push("## Your concerns");
    report.summary.answers.forEach((a) => {
      lines.push(`### ${a.concern}`);
      lines.push(a.answer);
      const cites = citeLine(a.citations);
      if (cites) lines.push(cites);
      lines.push("");
    });
  }

  if (report.summary.action_items.length) {
    lines.push("## Action items");
    sortedActionItems(report).forEach((item) => {
      const refs = item.photo_refs.length
        ? ` (photos: ${item.photo_refs.join(", ")})`
        : "";
      lines.push(`- [${item.priority}] ${item.text}${refs}`);
      const cites = citeLine(item.citations);
      if (cites) lines.push(`  - ${cites}`);
    });
    lines.push("");
  }

  lines.push("## Per-photo findings");
  report.per_photo.forEach((f) => {
    const name = photoNames[f.photo_id] ? ` (${photoNames[f.photo_id]})` : "";
    lines.push(`### ${f.photo_id}${name}`);
    lines.push(f.observation);
    const cites = citeLine(f.cited);
    if (cites) lines.push(cites);
    if (f.enrichment) {
      lines.push(
        `Fine-tuned model: ${f.enrichment.label} (${Math.round(
          f.enrichment.confidence * 100
        )}%, consistency-checked)`
      );
    }
    lines.push("");
  });

  const cardIds = Object.keys(report.cards || {});
  if (cardIds.length) {
    lines.push("## Cited guidance");
    cardIds.forEach((cid) => {
      const card = report.cards[cid];
      lines.push(`### ${card.title}`);
      lines.push(`- Passage: ${card.passage}`);
      lines.push(`- Citation: ${card.citation}`);
      lines.push(`- Source: ${card.source_url}`);
      lines.push("");
    });
  }

  const flagged = report.flagged_claims?.length || 0;
  if (flagged) {
    lines.push(flaggedLine(flagged));
    lines.push("");
  }
  return lines.join("\n");
}

function flaggedLine(n) {
  return n === 1
    ? "1 claim was dropped by the citation gate (no supporting guidance card)."
    : `${n} claims were dropped by the citation gate (no supporting guidance card).`;
}

function sortedActionItems(report) {
  return [...report.summary.action_items].sort(
    (a, b) => (PRIORITY_RANK[a.priority] ?? 1) - (PRIORITY_RANK[b.priority] ?? 1)
  );
}

function CiteChips({ ids, cards }) {
  if (!ids || !ids.length) return null;
  return (
    <span className="cite-chips">
      {ids.map((cid) => (
        <span key={cid} className="cite-chip" title={cards?.[cid]?.passage || cid}>
          {cards?.[cid]?.title || cid}
        </span>
      ))}
    </span>
  );
}

function Walkthrough({ API, pollMs = DEFAULT_POLL_MS }) {
  const [photos, setPhotos] = useState([]); // {file, preview, note}
  const [visitNote, setVisitNote] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const genRef = useRef(0); // supersede stale polls, like the analyze flow

  const addPhotos = (e) => {
    const incoming = Array.from(e.target.files || []);
    if (!incoming.length) return;
    setError("");
    setPhotos((prev) => {
      const room = MAX_WALKTHROUGH_PHOTOS - prev.length;
      if (incoming.length > room) setError(CAP_MESSAGE);
      const added = incoming.slice(0, Math.max(room, 0)).map((file) => ({
        file,
        preview: URL.createObjectURL(file),
        note: "",
      }));
      return [...prev, ...added];
    });
    e.target.value = ""; // allow re-adding the same file after a remove
  };

  const removePhoto = (index) => {
    genRef.current += 1;
    setReport(null);
    setPhotos((prev) => prev.filter((_, i) => i !== index));
  };

  const setPhotoNote = (index, value) => {
    setPhotos((prev) =>
      prev.map((p, i) => (i === index ? { ...p, note: value } : p))
    );
  };

  const handleSubmit = async () => {
    if (!photos.length) {
      setError("Add at least one photo first.");
      return;
    }
    const gen = ++genRef.current;
    const isCurrent = () => gen === genRef.current;
    setIsRunning(true);
    setError("");
    setReport(null);
    setStatus("Generating the diagnostic report...");

    const formData = new FormData();
    photos.forEach((p) => formData.append("files", p.file));
    photos.forEach((p) => formData.append("photo_notes", p.note || ""));
    if (visitNote.trim()) formData.append("visit_note", visitNote.trim());

    const submitJob = () =>
      axios.post(`${API}/walkthrough-jobs`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

    try {
      let submit;
      try {
        submit = await submitJob();
      } catch (err) {
        if (!isColdStartError(err)) throw err;
        setStatus("Model warming up - retrying...");
        await sleep(SUBMIT_RETRY_DELAY_MS);
        submit = await submitJob();
      }
      const jobId = submit.data.job_id;
      setStatus(
        "Reading the photos against the guidance corpus - a cold start can take a minute..."
      );

      let settled = false;
      for (let i = 0; i < MAX_POLLS && !settled; i++) {
        let poll;
        try {
          poll = await axios.get(`${API}/walkthrough-jobs/${jobId}`);
        } catch (pollErr) {
          if (pollErr?.response?.status === 500) throw pollErr; // worker failed
          if (i < MAX_POLLS - 1) await sleep(pollMs);
          continue;
        }
        if (poll.status === 200) {
          if (isCurrent()) setReport(poll.data);
          settled = true;
        } else if (i < MAX_POLLS - 1) {
          await sleep(pollMs); // 202 pending
        }
      }
      if (!settled && isCurrent()) setError(TIMEOUT_MESSAGE);
    } catch (err) {
      console.error("Error during walkthrough:", err);
      if (isCurrent()) {
        const statusCode = err?.response?.status;
        if (statusCode === 400 || statusCode === 413) {
          setError(err.response?.data?.detail || FAILED_MESSAGE);
        } else {
          setError(FAILED_MESSAGE);
        }
      }
    } finally {
      setIsRunning(false);
      setStatus("");
    }
  };

  const handleExport = () => {
    if (!report) return;
    const photoNames = {};
    photos.forEach((p, i) => {
      photoNames[`photo_${i + 1}`] = p.file.name;
    });
    const markdown = buildWalkthroughMarkdown(report, photoNames);
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const date = new Date().toISOString().slice(0, 10);
    const link = document.createElement("a");
    link.href = url;
    link.download = `walkthrough-report-${date}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const flaggedCount = report?.flagged_claims?.length || 0;
  const previewByPhotoId = {};
  photos.forEach((p, i) => {
    previewByPhotoId[`photo_${i + 1}`] = p.preview;
  });

  return (
    <section className="walkthrough-section">
      <h2>Walkthrough diagnostic report</h2>
      <p className="wt-subtitle">
        First site visit? Add the photos you took plus your concerns, and get a
        cited draft diagnostic: what is visible, what to check next, and what
        the photos cannot answer.
      </p>

      <input
        type="file"
        accept="image/*"
        multiple
        onChange={addPhotos}
        className="file-input"
        data-testid="wt-file-input"
        aria-label="Add walkthrough photos"
      />

      {photos.length > 0 && (
        <ul className="wt-photo-list">
          {photos.map((p, i) => (
            <li key={`${p.file.name}-${i}`} className="wt-photo-item" data-testid="wt-photo-item">
              <img src={p.preview} alt={`photo_${i + 1}`} className="wt-thumb" />
              <div className="wt-photo-meta">
                <span className="wt-photo-id">{`photo_${i + 1} - ${p.file.name}`}</span>
                <input
                  type="text"
                  className="wt-note-input"
                  data-testid="wt-photo-note"
                  placeholder="Optional note for this photo"
                  value={p.note}
                  maxLength={500}
                  onChange={(e) => setPhotoNote(i, e.target.value)}
                />
              </div>
              <button
                type="button"
                className="wt-remove-button"
                aria-label={`Remove photo ${i + 1}`}
                onClick={() => removePhoto(i)}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      <textarea
        className="note-input"
        placeholder="Your site-visit note: what is the client worried about? What did you see or smell? (drives the report's checklist)"
        value={visitNote}
        onChange={(e) => setVisitNote(e.target.value)}
        rows={3}
        maxLength={4000}
      />

      <button
        onClick={handleSubmit}
        disabled={isRunning}
        className="analyze-button"
      >
        {isRunning ? "Generating..." : "Generate report"}
      </button>
      {status && <p className="analyze-status">{status}</p>}
      {error && <div className="error-banner">{error}</div>}

      {report && (
        <div className="wt-report" data-testid="wt-report">
          <div className="wt-disclaimer">{report.disclaimer}</div>

          <h3>Overall assessment</h3>
          <p className="wt-assessment">{report.summary.overall_assessment}</p>
          {report.summary.assessment_citations?.length ? (
            <CiteChips ids={report.summary.assessment_citations} cards={report.cards} />
          ) : (
            <span className="wt-auto-summary-tag">
              auto-summary (derived from the cited findings)
            </span>
          )}

          {report.summary.answers.length > 0 && (
            <>
              <h3>Your concerns</h3>
              <ul className="wt-answers">
                {report.summary.answers.map((a) => (
                  <li
                    key={a.concern}
                    className={a.not_observed ? "wt-not-observed" : "wt-answer"}
                  >
                    <span className="wt-concern">{a.concern}</span>
                    <p className="wt-answer-text">{a.answer}</p>
                    <CiteChips ids={a.citations} cards={report.cards} />
                  </li>
                ))}
              </ul>
            </>
          )}

          {report.summary.action_items.length > 0 && (
            <>
              <h3>Action items</h3>
              <ul className="wt-actions">
                {sortedActionItems(report).map((item) => (
                  <li key={item.text} className="wt-action">
                    <span className={`priority-chip priority-${item.priority}`}>
                      {item.priority}
                    </span>
                    <span className="wt-action-text">{item.text}</span>
                    {item.photo_refs.length > 0 && (
                      <span className="wt-photo-refs">
                        {item.photo_refs.join(", ")}
                      </span>
                    )}
                    <CiteChips ids={item.citations} cards={report.cards} />
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3>Per-photo findings</h3>
          <ul className="wt-findings">
            {report.per_photo.map((f) => (
              <li
                key={f.photo_id}
                className={f.no_evidence ? "wt-finding wt-not-observed" : "wt-finding"}
              >
                {previewByPhotoId[f.photo_id] && (
                  <img
                    src={previewByPhotoId[f.photo_id]}
                    alt={f.photo_id}
                    className="wt-thumb"
                  />
                )}
                <div>
                  <span className="wt-photo-id">{f.photo_id}</span>
                  <p className="wt-observation">{f.observation}</p>
                  <CiteChips ids={f.cited} cards={report.cards} />
                  {f.enrichment && (
                    <span
                      className="wt-enrichment-chip"
                      title="Fine-tuned Qwen2.5-VL label, kept only when consistent with the observation"
                    >
                      {`fine-tuned: ${f.enrichment.label.replace(/_/g, " ")} ${Math.round(
                        f.enrichment.confidence * 100
                      )}%`}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>

          {flaggedCount > 0 && (
            <p className="wt-gate-note">{flaggedLine(flaggedCount)}</p>
          )}

          <button onClick={handleExport} className="export-button">
            Export walkthrough report (markdown)
          </button>
        </div>
      )}
    </section>
  );
}

export default Walkthrough;
