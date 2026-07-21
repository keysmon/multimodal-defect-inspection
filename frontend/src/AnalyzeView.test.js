import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import axios from "axios";
import AnalyzeView from "./AnalyzeView";

jest.mock("axios");

const API = "http://localhost:8000";

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

// Async /analyze: the submit POST returns 202 {job_id}; the first poll GET
// returns 200 with the full result body. Configures both mocks for one analysis.
function mockAnalyzeJob(resultResponse) {
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "job-async" } });
  axios.get.mockResolvedValueOnce({ status: 200, data: resultResponse.data });
}

beforeAll(() => {
  global.URL.createObjectURL = jest.fn(() => "blob:mock-url");
  global.URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  jest.clearAllMocks();
});

test("shows the result placeholder before any analysis", () => {
  render(<AnalyzeView API={API} />);
  expect(screen.getByText(/results appear here/i)).toBeInTheDocument();
});

test("analyze happy path shows severity headline, bars, and a guidance card", async () => {
  mockAnalyzeJob(mockAnalyzeResponse);
  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() =>
    expect(screen.getByTestId("severity-headline")).toHaveTextContent("Severity: Urgent")
  );
  expect(screen.getByTestId("rank-bars")).toHaveTextContent("crack");
  expect(screen.getByTestId("rank-bars")).toHaveTextContent("91%");

  // Guidance is collapsed by default; expanding reveals the card.
  expect(screen.queryByText("Assess crack width")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /show cited guidance \(1\)/i }));
  expect(screen.getByText("Assess crack width")).toBeInTheDocument();
});

test("strips markdown heading markers from the displayed description", async () => {
  const withHeading = JSON.parse(JSON.stringify(mockAnalyzeResponse));
  withHeading.data.description =
    "# Surface Condition Assessment\nA visible linear crack runs diagonally.";
  mockAnalyzeJob(withHeading);
  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() =>
    expect(screen.getByText(/Surface Condition Assessment/)).toBeInTheDocument()
  );
  expect(screen.queryByText(/# Surface Condition Assessment/)).not.toBeInTheDocument();
});

test("analyze posts the inspector note in the form data", async () => {
  mockAnalyzeJob(mockAnalyzeResponse);
  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.change(screen.getByLabelText(/inspector note/i), {
    target: { value: "musty smell near shower" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("note")).toBe("musty smell near shower");
});

test("analyze posts the selected audio file in the form data", async () => {
  mockAnalyzeJob(mockAnalyzeResponse);
  render(<AnalyzeView API={API} />);

  const img = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [img] },
  });
  const wav = new File(["RIFF"], "fan.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByTestId("audio-input"), {
    target: { files: [wav] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("audio")).toBe(wav);
});

test("analyze with audio shows the combined severity headline and audio panel", async () => {
  mockAnalyzeJob(mockAudioResponse);
  render(<AnalyzeView API={API} />);

  const img = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [img] },
  });
  const wav = new File(["RIFF"], "fan.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByTestId("audio-input"), {
    target: { files: [wav] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() =>
    expect(screen.getByTestId("severity-headline")).toHaveTextContent(
      "Combined severity: Structural"
    )
  );
  expect(screen.getByRole("heading", { name: /^equipment audio$/i })).toBeInTheDocument();
  expect(screen.getByText(/score: 0\.421/i)).toBeInTheDocument();
  expect(screen.getByText("anomalous")).toBeInTheDocument();
  expect(screen.getByText("Bearing wear rumble")).toBeInTheDocument();
});

test("selecting a new image clears a previously chosen audio file", () => {
  // Observable half of the stale-input fix: the chosen-audio filename display
  // disappears when a new image is picked. (The DOM input.value reset that
  // makes re-picking the SAME wav work is not observable in jsdom — file input
  // .value is always "" — so that half rides on the useRef idiom + Task 6 E2E.)
  render(<AnalyzeView API={API} />);

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
  render(<AnalyzeView API={API} />);

  const noteField = screen.getByLabelText(/inspector note/i);
  fireEvent.change(noteField, { target: { value: "old note" } });
  expect(noteField.value).toBe("old note");

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });

  expect(screen.getByLabelText(/inspector note/i).value).toBe("");
});

test("clicking a gallery example populates the note and runs the analyze flow", async () => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      blob: () =>
        Promise.resolve(new Blob(["img-bytes"], { type: "image/jpeg" })),
    })
  );
  mockAnalyzeJob(mockAnalyzeResponse);
  render(<AnalyzeView API={API} />);

  const tiles = screen.getAllByRole("button", { name: /load example:/i });
  expect(tiles).toHaveLength(6);
  fireEvent.click(tiles[0]);

  // The example's note is loaded into the inspector-note field...
  const noteField = screen.getByLabelText(/inspector note/i);
  await waitFor(() => expect(noteField.value).not.toBe(""));

  // ...and the analyze flow runs, posting that note in the form data.
  await waitFor(() => expect(axios.post).toHaveBeenCalled());
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("note")).toBe(noteField.value.trim());
  expect(formData.get("file")).toBeInstanceOf(File);

  await waitFor(() =>
    expect(screen.getByTestId("severity-headline")).toHaveTextContent("Severity: Urgent")
  );
});

test("auto-retries once on a cold-start timeout, then succeeds", async () => {
  jest.useFakeTimers();
  // First SUBMIT rejects (cold 504); the retry submits (202), then the first
  // poll returns the ready result.
  axios.post
    .mockRejectedValueOnce({ response: { status: 504 } })
    .mockResolvedValueOnce({ status: 202, data: { job_id: "job-async" } });
  axios.get.mockResolvedValueOnce({ status: 200, data: mockAnalyzeResponse.data });
  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  // First submit rejected -> warming status shown while the retry is pending.
  await act(async () => {});
  expect(screen.getByText(/Model warming up - retrying/i)).toBeInTheDocument();
  expect(axios.post).toHaveBeenCalledTimes(1);

  // Advance past the 3s backoff -> the single retry submits, then the poll runs.
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  await act(async () => {}); // flush the submit -> poll microtask chain
  expect(axios.post).toHaveBeenCalledTimes(2);
  expect(screen.getByTestId("severity-headline")).toHaveTextContent("Severity: Urgent");
  expect(
    screen.queryByText(/Model warming up - retrying/i)
  ).not.toBeInTheDocument();

  jest.useRealTimers();
});

test("shows an error banner when the analyze request fails", async () => {
  axios.post.mockRejectedValueOnce(new Error("network error"));
  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() =>
    expect(
      screen.getByText(/Analysis failed — is the API running\?/i)
    ).toBeInTheDocument()
  );
});

test("a mid-poll image swap does not render the superseded job's result", async () => {
  // HIGH race: analyze image A (cold -> pending poll), swap to image B via the
  // file input while A is still polling, then A's poll lands ready. A's result
  // must NOT render under B - it was superseded.
  jest.useFakeTimers();
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "jA" } });
  axios.get
    .mockResolvedValueOnce({ status: 202, data: { status: "pending" } }) // A poll 1
    .mockResolvedValueOnce({ status: 200, data: mockAnalyzeResponse.data }); // A poll 2 (superseded)
  render(<AnalyzeView API={API} />);

  const fileA = new File(["a"], "wallA.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [fileA] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));
  await act(async () => {}); // submit -> poll 1 (pending) -> sleep

  // Mid-poll: pick image B.
  const fileB = new File(["b"], "wallB.png", { type: "image/png" });
  await act(async () => {
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [fileB] } });
  });

  // Advance so A's second poll resolves ready.
  await act(async () => {
    jest.advanceTimersByTime(1500);
  });
  await act(async () => {});

  expect(screen.queryByTestId("severity-headline")).not.toBeInTheDocument();
  expect(screen.queryByText("Assess crack width")).not.toBeInTheDocument();
  jest.useRealTimers();
});

test("a 400 (unreadable image) shows the image error, not an API-down message", async () => {
  axios.post.mockRejectedValueOnce({
    response: { status: 400, data: { detail: "Uploaded file is not a readable image: broken" } },
  });
  render(<AnalyzeView API={API} />);

  const file = new File(["not-an-image"], "bad.txt", { type: "text/plain" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await waitFor(() =>
    expect(screen.getByText(/not a readable image/i)).toBeInTheDocument()
  );
  expect(
    screen.queryByText(/is the API running/i)
  ).not.toBeInTheDocument();
});

test("polls until the result is ready (202 pending, then 200)", async () => {
  jest.useFakeTimers();
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "j" } });
  axios.get
    .mockResolvedValueOnce({ status: 202, data: { status: "pending" } })
    .mockResolvedValueOnce({ status: 200, data: mockAnalyzeResponse.data });
  render(<AnalyzeView API={API} />);
  const file = new File(["x"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  await act(async () => {}); // submit + first poll (202) -> sleep
  await act(async () => {
    jest.advanceTimersByTime(1500);
  }); // second poll -> 200
  await act(async () => {});
  expect(screen.getByTestId("severity-headline")).toHaveTextContent("Severity: Urgent");
  expect(axios.get).toHaveBeenCalledTimes(2);
  jest.useRealTimers();
});

test("a terminal worker failure (poll 500) shows the analysis-failed message", async () => {
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "j" } });
  axios.get.mockRejectedValueOnce({ response: { status: 500 } });
  render(<AnalyzeView API={API} />);
  const file = new File(["x"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));
  await waitFor(() =>
    expect(screen.getByText(/Analysis failed\. Please try again/i)).toBeInTheDocument()
  );
});

test("a transient poll error keeps polling instead of aborting", async () => {
  jest.useFakeTimers();
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "j" } });
  axios.get
    .mockRejectedValueOnce({ response: { status: 503 } }) // transient
    .mockResolvedValueOnce({ status: 200, data: mockAnalyzeResponse.data });
  render(<AnalyzeView API={API} />);
  const file = new File(["x"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));
  await act(async () => {}); // submit + poll1 rejects 503 -> sleep
  await act(async () => {
    jest.advanceTimersByTime(1500);
  }); // poll2 -> 200
  await act(async () => {});
  expect(screen.getByTestId("severity-headline")).toHaveTextContent("Severity: Urgent");
  jest.useRealTimers();
});

test("gives up with a timeout message after the poll ceiling", async () => {
  jest.useFakeTimers();
  axios.post.mockResolvedValueOnce({ status: 202, data: { job_id: "j" } });
  axios.get.mockResolvedValue({ status: 202, data: { status: "pending" } }); // never ready
  render(<AnalyzeView API={API} />);
  const file = new File(["x"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));
  await act(async () => {}); // submit + first poll
  for (let i = 0; i < 95; i++) {
    await act(async () => {
      jest.advanceTimersByTime(1500);
    });
  }
  expect(screen.getByText(/taking longer than expected/i)).toBeInTheDocument();
  jest.useRealTimers();
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
    .mockResolvedValueOnce({ status: 202, data: { job_id: "job-async" } }) // /analyze-jobs submit
    .mockResolvedValueOnce(mockVlmSubmit); // /analyze-vlm submit
  axios.get
    .mockResolvedValueOnce({ status: 200, data: mockAnalyzeResponse.data }) // /analyze-jobs poll -> reveals GPU button
    .mockResolvedValueOnce(mockVlmReady); // /vlm-status ready on first poll

  render(<AnalyzeView API={API} />);

  const file = new File(["dummy-bytes"], "wall.png", { type: "image/png" });
  fireEvent.change(screen.getByTestId("file-input"), {
    target: { files: [file] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^analyze photo$/i }));

  // The GPU button only appears once there is an analysis result.
  const gpuButton = await screen.findByRole("button", {
    name: /re-run on the fine-tuned model/i,
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
  expect(screen.getByTestId("vlm-chips")).toHaveTextContent("1. exposed rebar 88%");
});
