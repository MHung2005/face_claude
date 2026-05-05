import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("onnxruntime-web", () => ({
  InferenceSession: { create: vi.fn() },
  env: { wasm: {} },
}));

import { cropFace } from "../ai";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AI faceUtils — cropFace", () => {
  it("exports mirrored crop bytes and mirrored local keypoints", async () => {
    const translate = vi.fn();
    const scale = vi.fn();
    const drawImage = vi.fn();
    const fakeBlob = new Blob(["face"], { type: "image/jpeg" });
    const originalCreateElement = document.createElement.bind(document);
    const fakeCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => ({
        drawImage,
        scale,
        translate,
      })),
      toBlob: vi.fn((callback) => callback(fakeBlob)),
    };

    vi.spyOn(document, "createElement").mockImplementation((tagName) => (
      tagName === "canvas" ? fakeCanvas : originalCreateElement(tagName)
    ));

    const result = await cropFace(
      { videoHeight: 80, videoWidth: 100 },
      {
        box: { x1: 10, y1: 20, x2: 50, y2: 60 },
        keypoints: [
          [15, 25],
          [45, 25],
          [30, 40],
          [20, 55],
          [40, 55],
        ],
      },
      0,
    );

    expect(translate).toHaveBeenCalledWith(40, 0);
    expect(scale).toHaveBeenCalledWith(-1, 1);
    expect(drawImage).toHaveBeenCalledWith(
      expect.objectContaining({ videoHeight: 80, videoWidth: 100 }),
      10,
      20,
      40,
      40,
      0,
      0,
      40,
      40,
    );
    expect(result.localKeypoints).toEqual([
      35, 5,
      5, 5,
      20, 20,
      30, 35,
      10, 35,
    ]);
    expect(result.blob).toBe(fakeBlob);
  });
});
