import { useCallback, useEffect, useRef, useState } from "react";
import { submitGuestCheckin } from "../lib/guestApi";
import { getFriendlyBackendErrorMessage } from "../lib/errorMessages";

// ── Constants ────────────────────────────────────────────────────────────────
const JETSON_FRAME_URL = "/jetson/frame";
const JETSON_SCAN_INTERVAL_MS = 1800; // grab a still every 1.8 s for recognition

// ── useJetsonRecognition ─────────────────────────────────────────────────────
// Periodically fetches a still JPEG from /jetson/frame, sends it to the
// existing /api/guest/checkin endpoint, and returns the latest result.
//
// State machine:
//   "disconnected"  — default; Jetson chưa từng phản hồi thành công
//   "scanning"      — đang gửi request (chỉ hiển thị sau khi đã kết nối)
//   "idle"          — request vừa xong, đang chờ interval tiếp theo
//   "error"         — đã từng kết nối nhưng bị mất liên lạc

export function useJetsonRecognition({ enabled }) {
  const [jetsonResult, setJetsonResult] = useState(null);
  const [jetsonStatus, setJetsonStatus] = useState("disconnected");

  const inflightRef = useRef(false);
  const timerRef = useRef(null);
  const statusRef = useRef("disconnected");
  const everConnectedRef = useRef(false);

  const setStatus = useCallback((next) => {
    if (statusRef.current === next) return;
    statusRef.current = next;
    setJetsonStatus(next);
  }, []);

  const runScan = useCallback(async () => {
    if (inflightRef.current) return;
    inflightRef.current = true;

    if (everConnectedRef.current) {
      setStatus("scanning");
    }

    try {
      const frameResp = await fetch(JETSON_FRAME_URL);
      if (!frameResp.ok) throw new Error(`Frame fetch failed: ${frameResp.status}`);
      const blob = await frameResp.blob();
      const file = new File([blob], "jetson-frame.jpg", { type: "image/jpeg" });

      const payload = await submitGuestCheckin(file);

      everConnectedRef.current = true;
      setJetsonResult(payload);
      setStatus("idle");
    } catch (err) {
      if (everConnectedRef.current) {
        setJetsonResult({
          status: "network_error",
          message: getFriendlyBackendErrorMessage(err, "Không thể kết nối Jetson camera."),
          checked_in_at: new Date().toISOString(),
        });
        setStatus("error");
      }
    } finally {
      inflightRef.current = false;
    }
  }, [setStatus]);

  const prevEnabledRef = useRef(enabled);
  useEffect(() => {
    if (!enabled) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      everConnectedRef.current = false;
      setStatus("disconnected");
      prevEnabledRef.current = enabled;
      return;
    }

    if (!prevEnabledRef.current && enabled) {
      everConnectedRef.current = false;
      setStatus("disconnected");
    }
    prevEnabledRef.current = enabled;

    if (!timerRef.current) {
      timerRef.current = setInterval(() => {
        void runScan();
      }, JETSON_SCAN_INTERVAL_MS);
      void runScan();
    }

    return () => {
      clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [enabled, runScan, setStatus]);

  return { jetsonResult, jetsonStatus, everConnectedRef };
}
