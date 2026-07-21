import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import App from "./App";

jest.mock("axios");

beforeAll(() => {
  global.URL.createObjectURL = jest.fn(() => "blob:mock-url");
  global.URL.revokeObjectURL = jest.fn();
});

test("renders the SiteCheck app bar with three tool tabs", () => {
  render(<App />);
  expect(screen.getByText("Site")).toBeInTheDocument();
  expect(screen.getByText("Check")).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "Walkthrough" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "Analyze" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "Search guidance" })).toBeInTheDocument();
  expect(screen.queryByText(/DefectLens/)).not.toBeInTheDocument();
});

test("walkthrough is the default tab; switching reveals the other panels", () => {
  render(<App />);
  expect(screen.getByRole("tabpanel", { name: "Walkthrough" })).toBeVisible();
  expect(screen.getByRole("tab", { name: "Walkthrough" })).toHaveAttribute("aria-selected", "true");

  fireEvent.click(screen.getByRole("tab", { name: "Analyze" }));
  expect(screen.getByRole("tabpanel", { name: "Analyze" })).toBeVisible();
  expect(document.getElementById("panel-walkthrough")).not.toBeVisible();

  fireEvent.click(screen.getByRole("tab", { name: "Search guidance" }));
  expect(screen.getByRole("tabpanel", { name: "Search guidance" })).toBeVisible();
});
