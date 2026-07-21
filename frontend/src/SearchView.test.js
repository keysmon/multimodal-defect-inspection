import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import axios from "axios";
import SearchView from "./SearchView";

jest.mock("axios");
afterEach(() => jest.clearAllMocks());

const API = "http://localhost:8000";
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

test("search happy path shows a guidance card and the count line", async () => {
  axios.post.mockResolvedValueOnce(mockSearchResponse);
  render(<SearchView API={API} />);
  fireEvent.change(screen.getByPlaceholderText(/musty smell/i), {
    target: { value: "moisture" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^search$/i }));
  await waitFor(() => expect(screen.getByText("Ventilation guidance")).toBeInTheDocument());
  expect(axios.post).toHaveBeenCalledWith(`${API}/search`, { query: "moisture" });
  expect(screen.getByText(/1 card · ranked by embedding similarity/i)).toBeInTheDocument();
});

test("empty query shows a validation error and does not call the API", () => {
  render(<SearchView API={API} />);
  fireEvent.click(screen.getByRole("button", { name: /^search$/i }));
  expect(screen.getByText(/Please enter a search query/i)).toBeInTheDocument();
  expect(axios.post).not.toHaveBeenCalled();
});
