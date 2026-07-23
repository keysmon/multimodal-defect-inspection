import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { CardList, ExemplarStrip, Lightbox } from "./ui";

const card = {
  id: "c1",
  title: "Assess crack width",
  passage: "Measure crack width using a comparator.",
  severity: "urgent",
  citation: "ACI 224R-01",
  source_name: "ACI",
  source_url: "https://example.com/aci",
};

const exemplar = {
  id: "mbdd-crack-001",
  thumb_url: "/exemplars/thumbs/mbdd-crack-001.jpg",
  image_url: "/exemplars/mbdd-crack-001.jpg",
  credit: "MBDD2025 building-defect dataset (Zenodo), CC BY 4.0",
  source_url: "https://zenodo.org/records/15622584",
  caption: "Facade crack, UAV survey crop",
};

test("CardList renders an exemplar thumb strip when the card carries exemplars", () => {
  render(<CardList cards={[{ ...card, exemplars: [exemplar] }]} />);

  const strip = screen.getByTestId("exemplar-strip");
  const thumb = strip.querySelector("button.sc-exemplar-thumb");
  expect(thumb).toHaveAttribute("title", exemplar.credit);
  expect(thumb.querySelector("img")).toHaveAttribute("src", exemplar.thumb_url);
});

test("CardList without exemplars renders no strip", () => {
  render(<CardList cards={[card]} />);
  expect(screen.queryByTestId("exemplar-strip")).not.toBeInTheDocument();
});

test("clicking an exemplar thumb opens the lightbox with full image and credit", () => {
  render(<ExemplarStrip exemplars={[exemplar]} />);

  fireEvent.click(screen.getByTitle(exemplar.credit));
  const lightbox = screen.getByTestId("exemplar-lightbox");
  expect(lightbox.querySelector("img")).toHaveAttribute("src", exemplar.image_url);
  expect(lightbox).toHaveTextContent(exemplar.credit);

  fireEvent.click(screen.getByRole("button", { name: /close/i }));
  expect(screen.queryByTestId("exemplar-lightbox")).not.toBeInTheDocument();
});

test("Lightbox closes on Escape and backdrop click", () => {
  const onClose = jest.fn();
  render(<Lightbox src="/x.jpg" alt="x" label="L" caption="C" onClose={onClose} />);

  fireEvent.keyDown(window, { key: "Escape" });
  expect(onClose).toHaveBeenCalledTimes(1);

  fireEvent.click(screen.getByTestId("sc-lightbox"));
  expect(onClose).toHaveBeenCalledTimes(2);
});
