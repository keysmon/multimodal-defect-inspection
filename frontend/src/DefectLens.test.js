import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
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

test("clicking a gallery example populates the note and runs the analyze flow", async () => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      blob: () =>
        Promise.resolve(new Blob(["img-bytes"], { type: "image/jpeg" })),
    })
  );
  axios.post.mockResolvedValueOnce(mockAnalyzeResponse);
  render(<DefectLens />);

  const tiles = screen.getAllByRole("button", { name: /load example:/i });
  expect(tiles).toHaveLength(6);
  fireEvent.click(tiles[0]);

  // The example's note is loaded into the inspector-note field...
  const noteField = screen.getByPlaceholderText(/optional inspector note/i);
  await waitFor(() => expect(noteField.value).not.toBe(""));

  // ...and the analyze flow runs, posting that note in the form data.
  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("note")).toBe(noteField.value.trim());
  expect(formData.get("file")).toBeInstanceOf(File);

  await waitFor(() =>
    expect(screen.getByText(/Severity: Urgent/i)).toBeInTheDocument()
  );
});

test("auto-retries once on a cold-start timeout, then succeeds", async () => {
  jest.useFakeTimers();
  axios.post
    .mockRejectedValueOnce({ response: { status: 504 } })
    .mockResolvedValueOnce(mockAnalyzeResponse);
  render(<DefectLens />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  // First attempt rejected -> warming status shown while the retry is pending.
  await act(async () => {});
  expect(screen.getByText(/Model warming up - retrying/i)).toBeInTheDocument();
  expect(axios.post).toHaveBeenCalledTimes(1);

  // Advance past the 3s backoff -> the single retry fires and succeeds.
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(axios.post).toHaveBeenCalledTimes(2);
  expect(screen.getByText(/Severity: Urgent/i)).toBeInTheDocument();
  expect(
    screen.queryByText(/Model warming up - retrying/i)
  ).not.toBeInTheDocument();

  jest.useRealTimers();
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

test("GPU button submits to the fine-tuned model and renders its result", async () => {
  const mockVlmSubmit = {
    data: {
      job_id: "job-1",
      output_location: "s3://b/async-out/job-1.out",
      failure_location: "s3://b/async-fail/job-1.out",
    },
  };
  // First poll returns "ready" (200), so no polling delay is exercised here.
  const mockVlmReady = {
    status: 200,
    data: {
      status: "ready",
      classes: [
        { label: "exposed_rebar", score: 0.88 },
        { label: "spalling", score: 0.09 },
        { label: "crack", score: 0.03 },
      ],
    },
  };
  axios.post
    .mockResolvedValueOnce(mockAnalyzeResponse) // /analyze reveals the GPU button
    .mockResolvedValueOnce(mockVlmSubmit); // /analyze-vlm submit
  axios.get.mockResolvedValueOnce(mockVlmReady); // /vlm-status ready on first poll

  render(<DefectLens />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));

  // The GPU button only appears once there is an analysis result.
  const gpuButton = await screen.findByRole("button", {
    name: /run fine-tuned model/i,
  });
  fireEvent.click(gpuButton);

  // It submits the image to /analyze-vlm...
  await waitFor(() =>
    expect(axios.post).toHaveBeenCalledWith(
      expect.stringContaining("/analyze-vlm"),
      expect.any(FormData),
      expect.anything()
    )
  );

  // ...polls /vlm-status with the returned output location...
  await waitFor(() =>
    expect(screen.getByText("fine-tuned VLM (GPU)")).toBeInTheDocument()
  );
  expect(axios.get).toHaveBeenCalledWith(
    expect.stringContaining("/vlm-status"),
    expect.objectContaining({
      params: expect.objectContaining({
        output_location: "s3://b/async-out/job-1.out",
      }),
    })
  );
  // ...and renders the fine-tuned model's top class.
  expect(screen.getByText("1. exposed rebar")).toBeInTheDocument();
});
