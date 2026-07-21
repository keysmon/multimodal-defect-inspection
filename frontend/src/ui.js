// Shared SiteCheck primitives, transcribed from the CertReps design-system
// bundle (Button, Pill) plus repo-specific helpers (severity mapping, cards).
import React from "react";

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
