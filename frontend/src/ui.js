// Shared SiteCheck primitives, transcribed from the CertReps design-system
// bundle (Button, Pill) plus repo-specific helpers (severity mapping, cards).
import React, { useEffect, useState } from "react";

export function Button({ variant = "primary", size = "md", children, style = {}, disabled, onClick, ...rest }) {
  const base = {
    fontFamily: "var(--font-sans)",
    fontWeight: 600,
    border: "none",
    borderRadius: size === "sm" ? "9px" : "11px",
    padding: size === "sm" ? "8px 14px" : "12px 20px",
    fontSize: size === "sm" ? "13px" : "14.5px",
    cursor: disabled ? "not-allowed" : "pointer",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "9px",
    textDecoration: "none",
    transition: ".14s",
    lineHeight: 1.2,
  };
  const variants = {
    primary: disabled
      ? { background: "var(--panel-3)", color: "var(--faint)", boxShadow: "none" }
      : { background: "var(--gold)", color: "var(--on-gold)", boxShadow: "var(--shadow-gold)" },
    ghost: { background: "transparent", color: "var(--text)", border: "1px solid var(--line-strong)" },
    danger: { background: "transparent", color: "var(--rose)", border: "1px solid var(--rose)" },
  };
  return (
    <button disabled={disabled} onClick={disabled ? undefined : onClick} style={{ ...base, ...variants[variant], ...style }} {...rest}>
      {children}
    </button>
  );
}

export function Pill({ tone = "default", children, title, style = {} }) {
  return (
    <span className={`sc-pill sc-pill--${tone}`} title={title} style={style}>
      {children}
    </span>
  );
}

// Severity band -> pill tone + display word (structural/urgent/monitor/cosmetic).
const SEVERITY_TONES = { structural: "bad", urgent: "urgent", monitor: "level", cosmetic: "good" };
export function severityTone(severity) {
  return SEVERITY_TONES[severity] || "default";
}
export function severityLabel(severity) {
  if (!severity) return "Unknown";
  return severity.charAt(0).toUpperCase() + severity.slice(1);
}

export function StatusLine({ children }) {
  if (!children) return null;
  return (
    <p className="sc-status">
      <span className="sc-status-dot" />
      <span>{children}</span>
    </p>
  );
}

export function ErrorBanner({ children }) {
  if (!children) return null;
  return <div className="sc-error">{children}</div>;
}

// Shared fullscreen photo lightbox — one implementation, two consumers
// (Walkthrough photo grid, exemplar thumbs). Esc and backdrop-click close.
export function Lightbox({ src, alt, label, caption, onClose, testId = "sc-lightbox" }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="sc-lightbox" data-testid={testId} onClick={onClose}>
      <div className="sc-lightbox-card" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="sc-lightbox-close" aria-label="Close" onClick={onClose}>
          ×
        </button>
        <img src={src} alt={alt} className="sc-lightbox-img" />
        {(label || caption) && (
          <div className="sc-lightbox-meta">
            {label && <span className="sc-lightbox-label">{label}</span>}
            {caption && <span className="sc-lightbox-obs">{caption}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

// Exemplar thumb strip (KB track): 44px documented-case thumbnails on a
// guidance card; hover shows the credit, click opens the shared lightbox.
export function ExemplarStrip({ exemplars }) {
  const [open, setOpen] = useState(null); // exemplar index or null
  if (!exemplars || exemplars.length === 0) return null;
  return (
    <>
      <div className="sc-exemplar-strip" data-testid="exemplar-strip">
        {exemplars.map((ex, i) => (
          <button
            key={ex.id}
            type="button"
            className="sc-exemplar-thumb"
            title={ex.credit}
            onClick={() => setOpen(i)}
          >
            <img src={ex.thumb_url} alt={ex.caption || ex.id} loading="lazy" />
          </button>
        ))}
      </div>
      {open != null && exemplars[open] && (
        <Lightbox
          src={exemplars[open].image_url}
          alt={exemplars[open].caption || exemplars[open].id}
          label={exemplars[open].credit}
          caption={exemplars[open].caption}
          onClose={() => setOpen(null)}
          testId="exemplar-lightbox"
        />
      )}
    </>
  );
}

// Shared guidance-card list for /analyze, /search and audio results.
export function CardList({ cards }) {
  if (!cards || cards.length === 0) return null;
  return (
    <ul className="sc-card-list">
      {cards.map((card) => (
        <li key={card.id} className="sc-card">
          <div className="sc-card-top">
            <h3 className="sc-card-title">{card.title}</h3>
            <Pill tone={severityTone(card.severity)}>{card.severity}</Pill>
          </div>
          <p className="sc-card-passage">{card.passage}</p>
          <ExemplarStrip exemplars={card.exemplars} />
          <div className="sc-card-cite">
            {card.id} · {card.citation} ·{" "}
            <a href={card.source_url} target="_blank" rel="noreferrer">
              {card.source_name}
            </a>
          </div>
        </li>
      ))}
    </ul>
  );
}
