// SiteCheck shell: app bar with three tool tabs. Views stay MOUNTED and are
// toggled with [hidden] so an in-flight analyze/walkthrough poll survives a
// tab switch instead of being unmounted mid-flight.
import React, { useState } from "react";
import Walkthrough from "./Walkthrough";
import DefectLens from "./DefectLens";
import SearchView from "./SearchView";
import "./theme.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

const TABS = [
  { id: "walkthrough", label: "Walkthrough" },
  { id: "analyze", label: "Analyze" },
  { id: "search", label: "Search guidance" },
];

function App() {
  const [tab, setTab] = useState("walkthrough");
  return (
    <div className="sc-app">
      <header className="sc-appbar">
        <div className="sc-appbar-inner">
          <div className="sc-wordmark">
            <span className="sc-wordmark-name">
              Site<span className="sc-gold">Check</span>
            </span>
            <span className="sc-wordmark-tag">inspection assistant</span>
          </div>
          <nav className="sc-tabs" role="tablist" aria-label="Tools">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={tab === t.id}
                aria-controls={`panel-${t.id}`}
                className="sc-tab"
                onClick={() => setTab(t.id)}
              >
                <span className="sc-tab-label">{t.label}</span>
              </button>
            ))}
          </nav>
          <span className="sc-appbar-badge">live demo · scales to zero</span>
        </div>
      </header>

      <div role="tabpanel" id="panel-walkthrough" aria-labelledby="tab-walkthrough" hidden={tab !== "walkthrough"}>
        <Walkthrough API={API} />
      </div>
      <div role="tabpanel" id="panel-analyze" aria-labelledby="tab-analyze" hidden={tab !== "analyze"}>
        <DefectLens />
      </div>
      <div role="tabpanel" id="panel-search" aria-labelledby="tab-search" hidden={tab !== "search"}>
        <SearchView API={API} />
      </div>

      <footer className="sc-footer">
        <span className="sc-footer-note">SiteCheck — always cited, verify before acting.</span>
      </footer>
    </div>
  );
}

export default App;
