// Walkthrough diagnostic report (P3): N photos + a visit note -> a grounded,
// cited initial-diagnostic report from the async /walkthrough-jobs path.
// SiteCheck presentation: scanning fills per-photo tag pills immediately; the
// full cited report is a client-side reveal ("Generate full summary") - the
// single /walkthrough-jobs job already returned it.
import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { isColdStartError, sleep } from "./apiHelpers";
import { Button, Pill, severityTone, StatusLine, ErrorBanner, Lightbox } from "./ui";

export const MAX_WALKTHROUGH_PHOTOS = 10;
const DEFAULT_POLL_MS = 1500;
// ~4.5 min ceiling: a cold worker loads models (~30s) then makes two Haiku
// calls over up to 10 photos; generous but bounded.
const MAX_POLLS = 180;
const SUBMIT_RETRY_DELAY_MS = 3000;

// GPU enrichment (fine-tuned model on the scale-to-zero SageMaker endpoint):
// user-triggered only; the endpoint sleeps at zero so the FIRST run pays a
// ~5 min cold start. Mirrors the single-photo GPU button's pacing.
const DEFAULT_ENRICH_POLL_MS = 10000;
const ENRICH_MAX_POLLS = 42; // ~7 min ceiling
const ENRICH_WARMING =
  "Fine-tuned model warming up on GPU - the first run can take ~5 minutes...";

const PRIORITY_RANK = { high: 0, medium: 1, low: 2 };
const CAP_MESSAGE = `A walkthrough is capped at ${MAX_WALKTHROUGH_PHOTOS} photos.`;
const FAILED_MESSAGE = "Walkthrough failed. Please try again.";
const TIMEOUT_MESSAGE =
  "The walkthrough is taking longer than expected. Please try again.";

// Worst cited-card severity for a photo's finding (photo-card urgency pill).
const SEV_RANK = { structural: 0, urgent: 1, monitor: 2, cosmetic: 3 };
function photoUrgency(finding, cards) {
  if (!finding || finding.no_evidence) return null;
  const sevs = (finding.cited || [])
    .map((cid) => cards?.[cid]?.severity)
    .filter(Boolean);
  if (!sevs.length) return null;
  return sevs.sort((a, b) => (SEV_RANK[a] ?? 9) - (SEV_RANK[b] ?? 9))[0];
}

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

// Citation id chips (mock style): the short card id, with the card's title +
// passage in the tooltip; the appendix renders the full card.
function CiteChips({ ids, cards }) {
  if (!ids || !ids.length) return null;
  return (
    <span className="sc-chip-row">
      {ids.map((cid) => (
        <span
          key={cid}
          className="sc-chip"
          title={cards?.[cid] ? `${cards[cid].title} - ${cards[cid].passage}` : cid}
        >
          {cid}
        </span>
      ))}
    </span>
  );
}

function EnrichmentPill({ enrichment }) {
  return (
    <Pill
      tone="good"
      title="Fine-tuned Qwen2.5-VL label, kept only when consistent with the observation"
    >
      {`fine-tuned: ${enrichment.label.replace(/_/g, " ")} ${Math.round(
        enrichment.confidence * 100
      )}%`}
    </Pill>
  );
}

function Walkthrough({
  API,
  pollMs = DEFAULT_POLL_MS,
  enrichPollMs = DEFAULT_ENRICH_POLL_MS,
}) {
  const [photos, setPhotos] = useState([]); // {file, preview, note}
  const [visitNote, setVisitNote] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [report, setReport] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState("");
  const [isEnriching, setIsEnriching] = useState(false);
  const [enrichStatus, setEnrichStatus] = useState("");
  const [enrichError, setEnrichError] = useState("");
  const [gate, setGate] = useState(null);
  const [showFull, setShowFull] = useState(false); // "Generate full summary" reveal
  const [lightbox, setLightbox] = useState(null); // photo index or null
  const genRef = useRef(0); // supersede stale polls, like the analyze flow
  const fileInputRef = useRef(null);

  // Blob-URL hygiene: previews are created per added photo; revoke them when
  // a photo is removed and when the component unmounts (photosRef mirrors
  // state so the unmount cleanup sees the latest list).
  const photosRef = useRef(photos);
  photosRef.current = photos;
  useEffect(
    () => () => photosRef.current.forEach((p) => URL.revokeObjectURL(p.preview)),
    []
  );

  const resetEnrich = () => {
    setIsEnriching(false);
    setEnrichStatus("");
    setEnrichError("");
    setGate(null);
  };

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
    setJobId(null);
    setShowFull(false);
    setLightbox(null);
    resetEnrich();
    setPhotos((prev) => {
      const removed = prev[index];
      if (removed) URL.revokeObjectURL(removed.preview);
      return prev.filter((_, i) => i !== index);
    });
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
    setJobId(null);
    setShowFull(false);
    resetEnrich();
    setStatus("Scanning the photos against the guidance corpus...");

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
      const submittedJobId = submit.data.job_id;
      setStatus(
        "Reading the photos against the guidance corpus - a cold start can take a minute..."
      );

      let settled = false;
      for (let i = 0; i < MAX_POLLS && !settled; i++) {
        let poll;
        try {
          poll = await axios.get(`${API}/walkthrough-jobs/${submittedJobId}`);
        } catch (pollErr) {
          if (pollErr?.response?.status === 500) throw pollErr; // worker failed
          if (i < MAX_POLLS - 1) await sleep(pollMs);
          continue;
        }
        if (poll.status === 200) {
          if (isCurrent()) {
            setReport(poll.data);
            setJobId(submittedJobId); // enables the GPU enrich action
          }
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

  const handleEnrich = async () => {
    if (!jobId || isEnriching) return;
    // Same supersede discipline as the submit flow: a photo removal or a new
    // report invalidates this enrich run; a late "ready" must not resurrect
    // a stale report onto the cleared/replaced UI.
    const gen = genRef.current;
    const isCurrent = () => gen === genRef.current;
    setIsEnriching(true);
    setEnrichError("");
    setGate(null);
    setEnrichStatus(ENRICH_WARMING);

    try {
      await axios.post(`${API}/walkthrough-jobs/${jobId}/enrich`);
      let settled = false;
      for (let i = 0; i < ENRICH_MAX_POLLS && !settled && isCurrent(); i++) {
        const poll = await axios.get(`${API}/walkthrough-jobs/${jobId}/enrich`);
        if (poll.data.status === "ready") {
          if (isCurrent()) {
            setReport(poll.data.report);
            setGate(poll.data.gate);
          }
          settled = true;
        } else {
          if (poll.data.done !== undefined && isCurrent()) {
            setEnrichStatus(
              `${ENRICH_WARMING} (${poll.data.done}/${poll.data.total} photos done)`
            );
          }
          if (i < ENRICH_MAX_POLLS - 1) await sleep(enrichPollMs);
        }
      }
      if (!settled && isCurrent()) {
        setEnrichError(
          "The fine-tuned model is taking longer than expected. Please try again."
        );
      }
    } catch (err) {
      console.error("Error during enrichment:", err);
      if (isCurrent()) {
        if (err?.response?.status === 503) {
          setEnrichError("The fine-tuned GPU model isn't deployed for this demo.");
        } else {
          setEnrichError("Enrichment failed. Please try again.");
        }
      }
    } finally {
      setIsEnriching(false);
      setEnrichStatus("");
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
  const findingByPhotoId = {};
  (report?.per_photo || []).forEach((f) => {
    findingByPhotoId[f.photo_id] = f;
  });
  const urgencies = (report?.per_photo || []).map((f) =>
    photoUrgency(f, report?.cards)
  );
  const urgentCount = urgencies.filter(
    (u) => u === "structural" || u === "urgent"
  ).length;
  const monitorCount = urgencies.filter(
    (u) => u === "monitor" || u === "cosmetic"
  ).length;
  const noEvidenceCount = (report?.per_photo || []).filter(
    (f) => f.no_evidence
  ).length;

  return (
    <main className="sc-main">
      <div className="sc-intro">
        <div className="sc-eyebrow">Walkthrough · site-visit report</div>
        <h1 className="sc-h1">Walk the site. Leave with a cited draft.</h1>
        <p className="sc-lede">
          Photos in, tags out — the full cited summary only when you ask.
        </p>
      </div>

      <div className="sc-layout">
        <aside className="sc-rail">
          <div className="sc-panel">
            <div className="sc-panel-row">
              <h2 className="sc-panel-title">This visit</h2>
              <span className="sc-panel-count">
                {photos.length} / {MAX_WALKTHROUGH_PHOTOS} photos
              </span>
            </div>
            <button
              type="button"
              className="sc-dashed-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              + Add photos
            </button>
            <input
              ref={fileInputRef}
              id="wt-file-input"
              type="file"
              accept="image/*"
              multiple
              onChange={addPhotos}
              className="sc-hidden-input"
              data-testid="wt-file-input"
              aria-label="Add walkthrough photos"
            />
            <div>
              <label className="sc-field-label" htmlFor="wt-visit-note">
                What's worrying you on this site?
              </label>
              <textarea
                id="wt-visit-note"
                className="sc-textarea"
                rows={4}
                maxLength={4000}
                placeholder="What is the client worried about? What did you see or smell? (drives the report's checklist)"
                value={visitNote}
                onChange={(e) => setVisitNote(e.target.value)}
              />
            </div>
            <Button onClick={handleSubmit} disabled={isRunning} style={{ width: "100%" }}>
              {isRunning ? "Scanning…" : "Scan photos"}
            </Button>
            <StatusLine>{status}</StatusLine>
            <ErrorBanner>{error}</ErrorBanner>
          </div>
        </aside>

        <div className="sc-results-col">
          {photos.length > 0 && (
            <div className="sc-photo-grid">
              {photos.map((p, i) => {
                const photoId = `photo_${i + 1}`;
                const finding = findingByPhotoId[photoId];
                const urgency = photoUrgency(finding, report?.cards);
                return (
                  <div
                    key={`${p.file.name}-${i}`}
                    className="sc-photo-card"
                    data-testid="wt-photo-item"
                  >
                    <img src={p.preview} alt={photoId} className="sc-photo-img" />
                    <button
                      type="button"
                      className="sc-photo-remove"
                      aria-label={`Remove photo ${i + 1}`}
                      onClick={() => removePhoto(i)}
                    >
                      ✕
                    </button>
                    <div className="sc-photo-foot">
                      <div className="sc-photo-foot-row">
                        <span className="sc-photo-label">
                          PHOTO {i + 1} · {p.file.name}
                        </span>
                        <button
                          type="button"
                          className="sc-photo-view"
                          onClick={() => setLightbox(i)}
                        >
                          VIEW
                        </button>
                        <span className="sc-photo-tags">
                          {finding?.no_evidence && (
                            <Pill tone="default">no evidence</Pill>
                          )}
                          {urgency && (
                            <Pill tone={severityTone(urgency)}>
                              {urgency === "structural" || urgency === "urgent"
                                ? "urgent"
                                : urgency}
                            </Pill>
                          )}
                          {(finding?.cited || []).slice(0, 2).map((cid) => (
                            <span
                              key={cid}
                              className="sc-chip"
                              title={report?.cards?.[cid]?.title || cid}
                            >
                              {cid}
                            </span>
                          ))}
                          {finding?.enrichment && (
                            <EnrichmentPill enrichment={finding.enrichment} />
                          )}
                        </span>
                      </div>
                      <input
                        type="text"
                        className="sc-text-input sc-photo-note"
                        data-testid="wt-photo-note"
                        placeholder="Optional note for this photo"
                        value={p.note}
                        maxLength={500}
                        onChange={(e) => setPhotoNote(i, e.target.value)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {report && (
            <div className="sc-strip" data-testid="wt-strip">
              <span
                className="sc-strip-dot"
                style={{
                  background: urgentCount
                    ? "var(--urgent)"
                    : noEvidenceCount === (report.per_photo || []).length
                      ? "var(--muted)"
                      : "var(--gold)",
                }}
              />
              <span className="sc-strip-text">
                {`${report.per_photo.length} photo${
                  report.per_photo.length === 1 ? "" : "s"
                } scanned - ${urgentCount} urgent, ${monitorCount} monitor, ${noEvidenceCount} no evidence`}
              </span>
              <div className="sc-strip-actions">
                <Button
                  variant={showFull ? "ghost" : "primary"}
                  onClick={() => setShowFull((s) => !s)}
                  style={{ height: 40 }}
                >
                  {showFull ? "Hide full summary" : "Generate full summary"}
                </Button>
              </div>
            </div>
          )}

          {report && showFull && (
            <article className="sc-article" data-testid="wt-report">
              <div className="sc-article-head">
                <div style={{ flex: 1, minWidth: 260 }}>
                  <div className="sc-article-meta">
                    {new Date().toISOString().slice(0, 10)} · {report.per_photo.length}{" "}
                    PHOTOS{visitNote.trim() ? " · 1 VISIT NOTE" : ""}
                  </div>
                  <h2 className="sc-article-title">Walkthrough diagnostic report</h2>
                  <span className="sc-disclaimer-pill">{report.disclaimer}</span>
                </div>
                <div className="sc-article-actions">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleEnrich}
                    disabled={isEnriching || !jobId}
                  >
                    {isEnriching
                      ? "Enriching with fine-tuned model…"
                      : "Enrich with fine-tuned model"}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={handleExport}>
                    Export markdown
                  </Button>
                </div>
              </div>

              {flaggedCount === 0 ? (
                <div className="sc-gate-banner">
                  <span className="sc-gate-dot" />
                  <span>
                    All claims cited —{" "}
                    <span className="sc-gate-mono">0 dropped by the gate</span>
                  </span>
                </div>
              ) : (
                <div className="sc-gate-banner sc-gate-banner--flagged">
                  <span className="sc-gate-dot sc-gate-dot--flagged" />
                  <span>{flaggedLine(flaggedCount)}</span>
                </div>
              )}

              <div className="sc-article-body">
                {(enrichStatus || enrichError) && (
                  <div>
                    <StatusLine>{enrichStatus}</StatusLine>
                    <ErrorBanner>{enrichError}</ErrorBanner>
                  </div>
                )}

                <section>
                  <h3 className="sc-section-h">Overall assessment</h3>
                  <p className="sc-prose">{report.summary.overall_assessment}</p>
                  {report.summary.assessment_citations?.length ? (
                    <CiteChips
                      ids={report.summary.assessment_citations}
                      cards={report.cards}
                    />
                  ) : (
                    <span className="sc-hint">
                      auto-summary (derived from the cited findings)
                    </span>
                  )}
                </section>

                {report.summary.answers.length > 0 && (
                  <section>
                    <h3 className="sc-section-h">Your concerns, answered</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                      {report.summary.answers.map((a) => (
                        <div
                          key={a.concern}
                          className={
                            a.not_observed
                              ? "sc-subcard sc-subcard--muted"
                              : "sc-subcard"
                          }
                        >
                          <div className="sc-subcard-q">{a.concern}</div>
                          <p className="sc-subcard-a">{a.answer}</p>
                          <div className="sc-chip-row">
                            {a.not_observed && (
                              <Pill tone="level">verify on-site</Pill>
                            )}
                            <CiteChips ids={a.citations} cards={report.cards} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                {report.summary.action_items.length > 0 && (
                  <section>
                    <h3 className="sc-section-h">What to do next</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      {sortedActionItems(report).map((item) => (
                        <div key={item.text} className="sc-action-row">
                          <Pill
                            tone={
                              item.priority === "high"
                                ? "bad"
                                : item.priority === "medium"
                                  ? "level"
                                  : "default"
                            }
                          >
                            {item.priority}
                          </Pill>
                          <div>
                            <p className="sc-action-text">{item.text}</p>
                            <CiteChips ids={item.citations} cards={report.cards} />
                          </div>
                          <span className="sc-action-refs">
                            {item.photo_refs.join(", ")}
                          </span>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                <section>
                  <h3 className="sc-section-h">Per-photo findings</h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {report.per_photo.map((f, i) => (
                      <div
                        key={f.photo_id}
                        className={
                          f.no_evidence ? "sc-finding sc-finding--muted" : "sc-finding"
                        }
                      >
                        <span className="sc-photo-badge">PHOTO {i + 1}</span>
                        <div>
                          <p className="sc-finding-text">{f.observation}</p>
                          <div className="sc-chip-row">
                            <CiteChips ids={f.cited} cards={report.cards} />
                            {f.enrichment && (
                              <EnrichmentPill enrichment={f.enrichment} />
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  {gate && (
                    <p className="sc-gate-note">
                      {`${gate.kept} label${gate.kept === 1 ? "" : "s"} merged, ${
                        gate.dropped.length
                      } dropped by the consistency gate`}
                    </p>
                  )}
                </section>

                {Object.keys(report.cards || {}).length > 0 && (
                  <section>
                    <h3 className="sc-section-h">Cited guidance</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {Object.entries(report.cards).map(([cid, card]) => (
                        <details key={cid} className="sc-details">
                          <summary>
                            {card.severity && (
                              <Pill tone={severityTone(card.severity)}>
                                {card.severity}
                              </Pill>
                            )}
                            <span className="sc-details-title">{card.title}</span>
                            <span className="sc-details-meta">
                              {cid}
                              {card.source_name ? ` · ${card.source_name}` : ""}
                            </span>
                          </summary>
                          <p className="sc-details-body">
                            {card.passage}{" "}
                            {card.source_url && (
                              <a href={card.source_url} target="_blank" rel="noreferrer">
                                {card.source_name || "Source"} →
                              </a>
                            )}
                          </p>
                        </details>
                      ))}
                    </div>
                  </section>
                )}
              </div>
            </article>
          )}
        </div>
      </div>

      {lightbox != null && photos[lightbox] && (
        <Lightbox
          src={photos[lightbox].preview}
          alt={`photo_${lightbox + 1}`}
          label={`PHOTO ${lightbox + 1} · ${photos[lightbox].file.name.toUpperCase()}`}
          caption={findingByPhotoId[`photo_${lightbox + 1}`]?.observation || ""}
          onClose={() => setLightbox(null)}
          testId="wt-lightbox"
        />
      )}
    </main>
  );
}

export default Walkthrough;
