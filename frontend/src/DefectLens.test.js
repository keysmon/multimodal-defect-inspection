import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import axios from "axios";
import DefectLens from "./DefectLens";

jest.mock("axios");

const mockAnalyzeResponse = {
  data: {
    classes: [
      { label: "crack", score: 0.91 },
      { label: "spalling", score: 0.82 },
      { label: "efflorescence", score: 0.71 },
      { label: "corrosion", score: 0.6 },
      { label: "delamination", score: 0.5 },
      { label: "settlement", score: 0.4 },
      { label: "moisture", score: 0.3 },
      { label: "staining", score: 0.2 },
      { label: "none", score: 0.1 },
    ],
    severity: "urgent",
    description: "Visible structural crack detected near the beam.",
    cards: [
      {
        id: "c1",
        title: "Assess crack width",
        passage: "Measure crack width using a comparator.",
        severity: "urgent",
        citation: "ACI 224R-01",
        source_name: "ACI",
        source_url: "https://example.com/aci",
      },
    ],
  },
};

const mockSearchResponse = {
  data: {
    cards: [
      {
        id: "s1",
        title: "Ventilation guidance",
        passage: "Ensure adequate airflow to reduce moisture buildup.",
        severity: "monitor",
        citation: "ASHRAE 62.2",
        source_name: "ASHRAE",
        source_url: "https://example.com/ashrae",
      },
    ],
  },
};

const mockAudioResponse = {
  data: {
    ...mockAnalyzeResponse.data,
    severity: "urgent",
    combined_severity: "structural",
    audio: {
      score: 0.4213,
      band: "anomalous",
      severity: "urgent",
      cards: [
        {
          id: "h1",
          title: "Bearing wear rumble",
          passage: "Low-frequency rumble that rises with load.",
          severity: "urgent",
          citation: "ASHRAE HVAC Applications",
          source_name: "ASHRAE",
          source_url: "https://example.com/ashrae-bearing",
        },
      ],
    },
  },
};

beforeAll(() => {
  global.URL.createObjectURL = jest.fn(() => "blob:mock-url");
  global.URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  jest.clearAllMocks();
});

test("renders the DefectLens header", () => {
  render(<DefectLens />);
  expect(
    screen.getByText(/DefectLens — building-defect inspection assistant/i)
  ).toBeInTheDocument();
});

test("analyze happy path shows severity banner and a guidance card", async () => {
  axios.post.mockResolvedValueOnce(mockAnalyzeResponse);
  render(<DefectLens />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  await waitFor(() =>
    expect(screen.getByText(/Severity: Urgent/i)).toBeInTheDocument()
  );
  expect(screen.getByText("Assess crack width")).toBeInTheDocument();
  expect(screen.getByText("1. crack")).toBeInTheDocument();
});

test("search happy path shows a guidance card", async () => {
  axios.post.mockResolvedValueOnce(mockSearchResponse);
  render(<DefectLens />);

  fireEvent.change(screen.getByPlaceholderText(/search defect guidance/i), {
    target: { value: "mold" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^search$/i }));

  await waitFor(() =>
    expect(screen.getByText("Ventilation guidance")).toBeInTheDocument()
  );
});

test("analyze posts the inspector note in the form data", async () => {
  axios.post.mockResolvedValueOnce(mockAnalyzeResponse);
  render(<DefectLens />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.change(screen.getByPlaceholderText(/optional inspector note/i), {
    target: { value: "musty smell near shower" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("note")).toBe("musty smell near shower");
});

test("analyze posts the selected audio file in the form data", async () => {
  axios.post.mockResolvedValueOnce(mockAnalyzeResponse);
  render(<DefectLens />);

  const img = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [img] },
  });
  const wav = new File(["RIFF"], "fan.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByTestId("audio-input"), {
    target: { files: [wav] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("audio")).toBe(wav);
});

test("analyze with audio shows the combined severity banner and audio panel", async () => {
  axios.post.mockResolvedValueOnce(mockAudioResponse);
  render(<DefectLens />);

  const img = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [img] },
  });
  const wav = new File(["RIFF"], "fan.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByTestId("audio-input"), {
    target: { files: [wav] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  await waitFor(() =>
    expect(
      screen.getByText(/Combined severity: Structural/i)
    ).toBeInTheDocument()
  );
  expect(screen.getByText("Equipment audio")).toBeInTheDocument();
  expect(screen.getByText(/score: 0\.421/i)).toBeInTheDocument();
  expect(screen.getByText("anomalous")).toBeInTheDocument();
  expect(screen.getByText("Bearing wear rumble")).toBeInTheDocument();
});

test("selecting a new image clears a previously chosen audio file", () => {
  // Observable half of the stale-input fix: the chosen-audio filename display
  // disappears when a new image is picked. (The DOM input.value reset that
  // makes re-picking the SAME wav work is not observable in jsdom — file input
  // .value is always "" — so that half rides on the useRef idiom + Task 7 E2E.)
  render(<DefectLens />);

  const img1 = new File(["a"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [img1] } });
  const wav = new File(["RIFF"], "fan.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByTestId("audio-input"), { target: { files: [wav] } });
  expect(screen.getByText("fan.wav")).toBeInTheDocument();

  const img2 = new File(["b"], "wall2.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [img2] } });
  expect(screen.queryByText("fan.wav")).not.toBeInTheDocument();
});

test("selecting a new image resets the inspector note", () => {
  render(<DefectLens />);

  const noteField = screen.getByPlaceholderText(/optional inspector note/i);
  fireEvent.change(noteField, { target: { value: "old note" } });
  expect(noteField.value).toBe("old note");

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });

  expect(screen.getByPlaceholderText(/optional inspector note/i).value).toBe("");
});

test("shows an error banner when the analyze request fails", async () => {
  axios.post.mockRejectedValueOnce(new Error("network error"));
  render(<DefectLens />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  await waitFor(() =>
    expect(
      screen.getByText(/Analysis failed — is the API running\?/i)
    ).toBeInTheDocument()
  );
});
