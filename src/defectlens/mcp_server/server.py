"""SiteCheck MCP server (stdio) - exposes the live inspection API as tools.

Config: SITECHECK_API_URL env (default = the public CloudFront API). Run via
the `sitecheck-mcp` console script. Requires `pip install defectlens[mcp]`.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from defectlens.mcp_server.client import SiteCheckClient

mcp = FastMCP("sitecheck")

_client_singleton: SiteCheckClient | None = None


def _client() -> SiteCheckClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = SiteCheckClient()
    return _client_singleton


@mcp.tool()
def analyze_photo(path: str, note: str = "") -> dict:
    """Analyze one building photo for defects.

    Returns ranked defect classes with probabilities, a rule-based severity,
    a condition description, cited guidance cards from the inspection-standards
    corpus, and visually similar documented cases. `path` is a local image
    file; `note` is an optional inspector note that steers retrieval.
    Takes up to ~60s when the cloud worker starts cold.
    """
    return _client().analyze_photo(path, note=note)


@mcp.tool()
def search_standards(query: str) -> dict:
    """Search the cited inspection-standards corpus (EPA/HUD/InterNACHI/FHWA).

    Free-text query -> guidance cards with citations and exemplar images.
    """
    return _client().search_standards(query)


@mcp.tool()
def run_walkthrough(
    photo_paths: list[str], visit_note: str = "", photo_notes: list[str] | None = None
) -> dict:
    """Generate a grounded, cited initial-diagnostic report from a site visit.

    Provide the walk's photos (local paths, up to the API's photo cap), an
    overall visit note describing the concerns, and optional per-photo notes
    (same order as photo_paths). Every claim in the report cites a real
    guidance card; unsupported claims come back as "not observed - verify
    on-site". Takes ~30-90s.
    """
    return _client().run_walkthrough(
        photo_paths, visit_note=visit_note, photo_notes=photo_notes
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
