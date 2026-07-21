// Search the cited guidance corpus directly (POST /search).
import React, { useState } from "react";
import axios from "axios";
import { Button, CardList, ErrorBanner } from "./ui";

function SearchView({ API }) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError("Please enter a search query.");
      return;
    }
    setIsSearching(true);
    setError("");
    try {
      const response = await axios.post(`${API}/search`, { query: searchQuery });
      setSearchResult({ cards: response.data.cards });
    } catch (err) {
      console.error("Error during search:", err);
      setError("Search failed — is the API running?");
      setSearchResult(null);
    } finally {
      setIsSearching(false);
    }
  };

  const count = searchResult?.cards?.length ?? null;
  return (
    <main className="sc-main sc-main--narrow">
      <div className="sc-intro">
        <div className="sc-eyebrow">Search · guidance corpus</div>
        <h1 className="sc-h1">Ask the standards directly.</h1>
        <p className="sc-lede">205 cited cards from EPA, HUD, InterNACHI, FHWA and NPS.</p>
      </div>
      <div className="sc-search-row">
        <input
          type="text"
          className="sc-search-input"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="e.g. musty smell behind paneling, grinding pump bearing…"
          aria-label="Search the guidance corpus"
        />
        <Button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? "Searching…" : "Search"}
        </Button>
      </div>
      <ErrorBanner>{error}</ErrorBanner>
      {count !== null && (
        <div className="sc-count-line">
          {count} card{count === 1 ? "" : "s"} · ranked by embedding similarity
        </div>
      )}
      <CardList cards={searchResult?.cards} />
    </main>
  );
}

export default SearchView;
