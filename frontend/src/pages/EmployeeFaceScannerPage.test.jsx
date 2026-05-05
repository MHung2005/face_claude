import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

import { renderApp } from "../test/test-utils";

vi.mock("../ai", () => ({
  cropFace: vi.fn(),
  detectFaces: vi.fn().mockResolvedValue([]),
  isModelLoaded: vi.fn(() => true),
  loadModel: vi.fn().mockResolvedValue(undefined),
}));

function mockJsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Employee face scanner page", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });

    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }],
        }),
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the scanner page with goal-based capture progress", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(mockJsonResponse({ manager: { id: 1, username: "manager" } }))
        .mockResolvedValueOnce(
          mockJsonResponse({
            employee: { id: 1, employee_code: "NV001", full_name: "Nguyen Van A", department: "Accounting" },
            face_samples: [],
            capture_config: {
              min_frames: 20,
              max_frames: 30,
              thumbnail_limit: 10,
              min_capture_gap_ms: 700,
            },
          }),
        ),
    );

    const { container } = renderApp("/manager/employees/1/face-registration");

    expect(await screen.findByText(/mã nhân viên: nv001/i)).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /đăng ký khuôn mặt/i })).toBeInTheDocument();
    expect(await screen.findByText(/xem trước thu thập/i)).toBeInTheDocument();
    expect(await screen.findByText(/0 \/ 20 ảnh đã nhận/i)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /bắt đầu thu thập \(20 ảnh\)/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /quản lý ảnh tĩnh/i })).toBeInTheDocument();
    expect(container.querySelector(".simple-scanner-video--mirrored")).toBeInTheDocument();
  });
});
