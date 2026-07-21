import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import axios from "axios";
import Walkthrough, {
  MAX_WALKTHROUGH_PHOTOS,
  buildWalkthroughMarkdown,
} from "./Walkthrough";

jest.mock("axios");

beforeAll(() => {
  global.URL.createObjectURL = jest.fn(() => "blob:preview");
  global.URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  jest.clearAllMocks();
});

const API = "http://localhost:8000";

function makeFile(name) {
  return new File(["img-bytes"], name, { type: "image/png" });
}

const mockReport = {
  concerns: ["is the crack active?", "what should be budgeted?"],
  per_photo: [
    {
      photo_id: "photo_1",
      observation: "hairline crack at sill",
      cited: ["crack-01"],
      no_evidence: false,
      enrichment: null,
    },
    {
      photo_id: "photo_2",
      observation:
        "Not observed - no defect matched to guidance in this photo; verify on-site.",
      cited: [],
      no_evidence: true,
      enrichment: null,
    },
  ],
  summary: {
    overall_assessment: "One active-looking crack across photos.",
    assessment_citations: ["crack-01"],
    action_items: [
      {
        priority: "high",
        text: "Measure the crack width",
        citations: ["crack-01"],
        photo_refs: ["photo_1"],
      },
    ],
    answers: [
      {
        concern: "is the crack active?",
        answer: "Monitor the width over two weeks",
        citations: ["crack-01"],
        not_observed: false,
      },
      {
        concern: "what should be budgeted?",
        answer: "Not observed in these photos - verify on-site.",
        citations: [],
        not_observed: true,
      },
    ],
  },
  disclaimer: "Initial diagnostic - verify before acting.",
  flagged_claims: [{ text: "invented", reason: "no_valid_citation" }],
  cards: {
    "crack-01": {
      id: "crack-01",
      title: "Crack width assessment",
      passage: "Measure with a comparator.",
      severity: "monitor",
      citation: "ACI 224R",
      source_name: "ACI",
      source_url: "https://example.com/aci",
    },
  },
};

function addPhotos(n, names) {
  const input = screen.getByTestId("wt-file-input");
  const files = Array.from({ length: n }, (_, i) =>
    makeFile(names ? names[i] : `p${i + 1}.png`)
  );
  fireEvent.change(input, { target: { files } });
}

test("renders the picker, caps photos at the walkthrough limit", () => {
  render(<Walkthrough API={API} />);
  expect(screen.getByTestId("wt-file-input")).toBeInTheDocument();
  expect(
    screen.getByPlaceholderText(/site-visit note/i)
  ).toBeInTheDocument();

  addPhotos(MAX_WALKTHROUGH_PHOTOS + 1);
  expect(screen.getByText(/capped at 10 photos/i)).toBeInTheDocument();
  expect(screen.getAllByTestId("wt-photo-item")).toHaveLength(
    MAX_WALKTHROUGH_PHOTOS
  );
});

test("submits files + notes and renders the polled report", async () => {
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "wt-1" } });
  axios.get
    .mockResolvedValueOnce({ status: 202, data: { status: "pending" } })
    .mockResolvedValueOnce({ status: 200, data: mockReport });

  render(<Walkthrough API={API} pollMs={0} />);
  addPhotos(2, ["a.png", "b.png"]);
  fireEvent.change(screen.getByPlaceholderText(/site-visit note/i), {
    target: { value: "is the crack active?" },
  });
  fireEvent.change(screen.getAllByTestId("wt-photo-note")[0], {
    target: { value: "near sill" },
  });
  fireEvent.click(screen.getByRole("button", { name: /generate report/i }));

  await waitFor(() =>
    expect(screen.getByText(/One active-looking crack/i)).toBeInTheDocument()
  );

  // request shape
  expect(axios.post).toHaveBeenCalledWith(
    `${API}/walkthrough-jobs`,
    expect.any(FormData),
    expect.anything()
  );
  const formData = axios.post.mock.calls[0][1];
  expect(formData.getAll("files")).toHaveLength(2);
  expect(formData.get("visit_note")).toBe("is the crack active?");
  expect(formData.getAll("photo_notes")).toEqual(["near sill", ""]);

  // rendered report pieces
  expect(screen.getByText(/Initial diagnostic - verify before acting/i)).toBeInTheDocument();
  expect(screen.getByText(/hairline crack at sill/i)).toBeInTheDocument();
  expect(screen.getByText(/Measure the crack width/i)).toBeInTheDocument();
  expect(screen.getByText(/Monitor the width over two weeks/i)).toBeInTheDocument();
  // the not-observed answer is styled as such
  const notObserved = screen.getAllByText(/verify on-site/i);
  expect(notObserved.length).toBeGreaterThan(0);
  expect(document.querySelector(".wt-not-observed")).not.toBeNull();
  // citations resolve to card titles
  expect(screen.getAllByText(/Crack width assessment/i).length).toBeGreaterThan(0);
  // gate activity surfaced honestly
  expect(screen.getByText(/1 claim was dropped by the citation gate/i)).toBeInTheDocument();
});

test("worker failure surfaces the generic error", async () => {
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "wt-2" } });
  axios.get.mockRejectedValueOnce({ response: { status: 500 } });

  render(<Walkthrough API={API} pollMs={0} />);
  addPhotos(1);
  fireEvent.click(screen.getByRole("button", { name: /generate report/i }));

  await waitFor(() =>
    expect(
      screen.getByText(/walkthrough failed\. please try again\./i)
    ).toBeInTheDocument()
  );
});

test("buildWalkthroughMarkdown includes citations, disclaimer and sources", () => {
  const md = buildWalkthroughMarkdown(mockReport, { photo_1: "a.png", photo_2: "b.png" });
  expect(md).toContain("# Walkthrough diagnostic report");
  expect(md).toContain("Initial diagnostic - verify before acting.");
  expect(md).toContain("## Overall assessment");
  expect(md).toContain("One active-looking crack");
  expect(md).toContain("Crack width assessment"); // citation resolved to title
  expect(md).toContain("Not observed in these photos - verify on-site.");
  expect(md).toContain("[high] Measure the crack width");
  expect(md).toContain("photo_1 (a.png)");
  expect(md).toContain("https://example.com/aci");
  expect(md).toContain("1 claim was dropped by the citation gate");
});

async function renderWithReport() {
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "wt-9" } });
  axios.get.mockResolvedValueOnce({ status: 200, data: mockReport });
  render(<Walkthrough API={API} pollMs={0} enrichPollMs={0} />);
  addPhotos(2, ["a.png", "b.png"]);
  fireEvent.click(screen.getByRole("button", { name: /generate report/i }));
  await waitFor(() =>
    expect(screen.getByTestId("wt-report")).toBeInTheDocument()
  );
}

test("enrich button submits and merges the gated GPU labels", async () => {
  await renderWithReport();

  const enriched = JSON.parse(JSON.stringify(mockReport));
  enriched.per_photo[0].enrichment = {
    label: "spalling",
    confidence: 0.82,
    consistent: true,
  };
  axios.post.mockResolvedValueOnce({
    status: 202,
    data: { status: "submitted", photos: 2 },
  });
  axios.get
    .mockResolvedValueOnce({ status: 202, data: { status: "pending", done: 1, total: 2 } })
    .mockResolvedValueOnce({
      status: 200,
      data: {
        status: "ready",
        report: enriched,
        gate: {
          kept: 1,
          dropped: [
            { photo_id: "photo_2", label: "spalling", confidence: 0.95,
              reason: "inconsistent_with_observation" },
          ],
        },
      },
    });

  fireEvent.click(
    screen.getByRole("button", { name: /enrich with fine-tuned model/i })
  );

  await waitFor(() =>
    expect(screen.getByText(/fine-tuned: spalling 82%/i)).toBeInTheDocument()
  );
  expect(axios.post).toHaveBeenLastCalledWith(`${API}/walkthrough-jobs/wt-9/enrich`);
  expect(
    screen.getByText(/1 label merged, 1 dropped by the consistency gate/i)
  ).toBeInTheDocument();
});

test("enrich surfaces the GPU-not-deployed message on 503", async () => {
  await renderWithReport();
  axios.post.mockRejectedValueOnce({ response: { status: 503 } });

  fireEvent.click(
    screen.getByRole("button", { name: /enrich with fine-tuned model/i })
  );
  await waitFor(() =>
    expect(
      screen.getByText(/fine-tuned GPU model isn't deployed/i)
    ).toBeInTheDocument()
  );
});
