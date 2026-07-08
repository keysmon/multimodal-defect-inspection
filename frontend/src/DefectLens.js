// src/DefectLens.js
import React, { useState } from "react";
import axios from "axios";
import "./DefectLens.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

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

  return lines.join("\n");
}

function DefectLens() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);

  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file || null);
    setImagePreview(file ? URL.createObjectURL(file) : null);
    setAnalyzeResult(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }
    setIsAnalyzing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAnalyzeResult({ ...response.data, filename: selectedFile.name });
    } catch (err) {
      console.error("Error during analyze:", err);
      setError("Analysis failed — is the API running?");
      setAnalyzeResult(null);
    } finally {
      setIsAnalyzing(false);
    }
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
  const bandStyle = analyzeResult ? severityStyle(analyzeResult.severity) : null;

  return (
    <div className="defectlens-container">
      <header className="dl-header">
        <h1 className="dl-title">
          DefectLens — building-defect inspection assistant
        </h1>
      </header>

      {error && <div className="error-banner">{error}</div>}

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
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          className="analyze-button"
        >
          {isAnalyzing ? "Analyzing..." : "Analyze"}
        </button>
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
            Severity: {bandStyle.label}
          </div>

          <div className="rank-chips">
            {topClasses.map((c, i) => (
              <span key={c.label} className="rank-chip">
                {`${i + 1}. ${c.label}`}
              </span>
            ))}
          </div>

          {analyzeResult.description && (
            <p className="description">{analyzeResult.description}</p>
          )}

          <CardList cards={analyzeResult.cards} />
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
