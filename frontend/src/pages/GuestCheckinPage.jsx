import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useGuestCamera } from "../hooks/useGuestCamera";
import { useYoloDetection } from "../hooks/useYoloDetection";
import { submitGuestCheckin, submitGuestCheckinKpts, waitGuestCheckinTaskResult } from "../lib/guestApi";
import { getFriendlyBackendErrorMessage, getGuestResultCopy } from "../lib/errorMessages";
import "./GuestCheckinPage.css";

// ── Jetson camera constants ──────────────────────────────────────────────────
const JETSON_STREAM_URL = "/jetson/stream";
const JETSON_FRAME_URL  = "/jetson/frame";
const JETSON_SCAN_INTERVAL_MS = 1800; // grab a still every 1.8 s for recognition

const MAX_HISTORY_ITEMS = 10;
const CHECKIN_COOLDOWN_MS = 60000;

const BOX_COLORS = {
  detecting:   "#00e5ff",
  recognizing: "#ffa726",
  recognized:  "#00FF00",
  unknown:     "#ef5350",
};

function getTone(status) {
  if (status === "recognized" || status === "already_checked_in") return "success";
  if (status === "multiple_faces") return "warning";
  if (status === "network_error" || status === "unknown") return "danger";
  return "scanning";
}

function getStatusLabel(cameraState) {
  if (cameraState !== "ready") return "Lỗi camera";
  return "Đang quét";
}

function getConfidenceValue(distance) {
  if (distance == null || Number.isNaN(distance)) return 0;
  return Math.max(0, Math.min(100, Math.round((1 - distance) * 1000) / 10));
}

function getEmployeeInitials(name) {
  if (!name) return "AI";
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("");
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--:--";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDateTime(value) {
  if (!value) return "Đang chờ dữ liệu";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Đang chờ dữ liệu";
  return `${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${date.toLocaleDateString()}`;
}

function getHistoryBadge(entry) {
  if (entry.status === "recognized" || entry.status === "already_checked_in") return "badge-success";
  if (entry.status === "multiple_faces" || entry.status === "no_face") return "badge-warning";
  return "badge-error";
}

// ── JetsonStream ─────────────────────────────────────────────────────────────
// Tách thành component riêng + React.memo để MJPEG <img> không bị remount
// mỗi khi parent re-render. Khi Jetson không kết nối được, browser sẽ retry
// liên tục gây broken-image flicker — onError dùng exponential backoff để
// giảm tần suất retry thay vì để browser tự retry ngay lập tức.
const JetsonStream = React.memo(function JetsonStream({ streamUrl }) {
  const imgRef      = useRef(null);
  const retryTimer  = useRef(null);
  const retryDelay  = useRef(2000); // bắt đầu 2s, tăng dần tới 30s

  const scheduleRetry = useCallback(() => {
    if (retryTimer.current) return; // đã có timer đang chờ
    retryTimer.current = setTimeout(() => {
      retryTimer.current = null;
      if (imgRef.current) {
        // Gán lại src để browser thử load lại
        imgRef.current.src = `${streamUrl}?t=${Date.now()}`;
      }
      // Tăng delay theo exponential backoff, tối đa 30s
      retryDelay.current = Math.min(retryDelay.current * 2, 30000);
    }, retryDelay.current);
  }, [streamUrl]);

  const handleLoad = useCallback(() => {
    // Kết nối thành công — reset backoff
    retryDelay.current = 2000;
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, []);

  return (
    <img
      ref={imgRef}
      src={streamUrl}
      alt="Jetson camera stream"
      className="kiosk-video"
      style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
      onLoad={handleLoad}
      onError={scheduleRetry}
    />
  );
});

function PerfHud({ getPerfSnapshot }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const iv = setInterval(() => {
      if (!containerRef.current || !getPerfSnapshot) return;
      const p = getPerfSnapshot();
      const d = p.detect || {};
      const b = p.backend || {};
      const lines = [
        `YOLO Detect   : ${d.total_ms ?? "—"}ms`,
        `  Preprocess  : ${d.preprocess_ms ?? "—"}ms`,
        `  Inference   : ${d.inference_ms ?? "—"}ms`,
        `  NMS         : ${d.postprocess_ms ?? "—"}ms`,
        `Crop+Encode   : ${p.crop ?? "—"}ms`,
        `Network (API) : ${p.network ?? "—"}ms`,
        `  Backend     : ${b.total_ms ?? "—"}ms`,
        `    Align     : ${b.align_ms ?? "—"}ms`,
        `    ArcFace   : ${b.get_feat_ms ?? "—"}ms`,
        `    KNN       : ${b.knn_ms ?? "—"}ms`,
        `    DB Write  : ${b.db_ms ?? "—"}ms`,
      ];
      containerRef.current.textContent = lines.join("\n");
    }, 500);
    return () => clearInterval(iv);
  }, [getPerfSnapshot]);

  return <pre ref={containerRef} className="perf-hud">Loading...</pre>;
}

// ── useJetsonRecognition ─────────────────────────────────────────────────────
// Periodically fetches a still JPEG from /jetson/frame, sends it to the
// existing /api/guest/checkin endpoint, and returns the latest result.
//
// State machine:
//   "disconnected"  — default; Jetson chưa từng phản hồi thành công
//   "scanning"      — đang gửi request (chỉ hiển thị sau khi đã kết nối)
//   "idle"          — request vừa xong, đang chờ interval tiếp theo
//   "error"         — đã từng kết nối nhưng bị mất liên lạc
//
// Nguyên tắc: UI chỉ thay đổi khi state thực sự thay đổi (statusRef guard).
// Trước khi có response thành công đầu tiên, mọi lỗi đều im lặng — hook
// tiếp tục retry trong nền mà không làm giật UI.

function useJetsonRecognition({ enabled }) {
  const [jetsonResult, setJetsonResult] = useState(null);
  // "disconnected" là trạng thái mặc định — chưa kết nối lần nào
  const [jetsonStatus, setJetsonStatus] = useState("disconnected");

  const inflightRef      = useRef(false);
  const timerRef         = useRef(null);
  const statusRef        = useRef("disconnected"); // shadow để dedup setState
  const everConnectedRef = useRef(false);           // đã có response thành công chưa

  // Chỉ gọi setJetsonStatus khi giá trị thực sự thay đổi
  const setStatus = useCallback((next) => {
    if (statusRef.current === next) return;
    statusRef.current = next;
    setJetsonStatus(next);
  }, []);

  const runScan = useCallback(async () => {
    if (inflightRef.current) return;
    inflightRef.current = true;

    // Chỉ báo "scanning" nếu đã từng kết nối thành công — tránh
    // nhấp nháy "scanning → disconnected → scanning" khi chưa có gì
    if (everConnectedRef.current) {
      setStatus("scanning");
    }

    try {
      const frameResp = await fetch(JETSON_FRAME_URL);
      if (!frameResp.ok) throw new Error(`Frame fetch failed: ${frameResp.status}`);
      const blob = await frameResp.blob();
      const file  = new File([blob], "jetson-frame.jpg", { type: "image/jpeg" });

      const payload = await submitGuestCheckin(file);

      // Lần đầu kết nối thành công — mở khoá hiển thị trạng thái
      everConnectedRef.current = true;
      setJetsonResult(payload);
      setStatus("idle");
    } catch (err) {
      if (everConnectedRef.current) {
        // Đã từng kết nối → báo lỗi để user biết bị mất liên lạc
        setJetsonResult({
          status: "network_error",
          message: getFriendlyBackendErrorMessage(err, "Không thể kết nối Jetson camera."),
          checked_in_at: new Date().toISOString(),
        });
        setStatus("error");
      }
      // Chưa từng kết nối → im lặng, giữ "disconnected", retry tiếp
    } finally {
      inflightRef.current = false;
    }
  }, [setStatus]);

  // Đảm bảo chỉ reset và khởi động interval khi enabled chuyển từ false -> true
  const prevEnabledRef = useRef(enabled);
  useEffect(() => {
    // Nếu tắt Jetson mode
    if (!enabled) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      // Reset toàn bộ khi tắt Jetson mode
      everConnectedRef.current = false;
      setStatus("disconnected");
      prevEnabledRef.current = enabled;
      return;
    }

    // Chỉ reset khi enabled chuyển từ false -> true
    if (!prevEnabledRef.current && enabled) {
      everConnectedRef.current = false;
      setStatus("disconnected");
    }
    prevEnabledRef.current = enabled;

    // Luôn chỉ tạo interval một lần khi enabled true
    if (!timerRef.current) {
      timerRef.current = setInterval(() => {
        void runScan();
      }, JETSON_SCAN_INTERVAL_MS);
      void runScan();
    }

    // Cleanup interval khi unmount hoặc khi enabled chuyển về false
    return () => {
      clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [enabled, runScan, setStatus]);

  return { jetsonResult, jetsonStatus };
}

// ── Main page component ──────────────────────────────────────────────────────
export default function GuestCheckinPage() {
  // Camera source toggle: "webcam" | "jetson"
  const [cameraSource, setCameraSource] = useState("webcam");
  const isJetson = cameraSource === "jetson";

  const {
    videoRef,
    cameraState,
    cameraError,
    retryCamera,
    stopCamera,
    cameraDevices = [],
    selectedCameraId = "",
    selectCamera,
  } = useGuestCamera();

  const [submissionState, setSubmissionState] = useState("idle");
  const [result, setResult]                   = useState(null);
  const [history, setHistory]                 = useState([]);
  const [manualFile, setManualFile]           = useState(null);
  const [showFallback, setShowFallback]       = useState(false);
  const [statusText, setStatusText]           = useState("AI đang quét khuôn mặt theo thời gian thực.");
  const overlayCanvasRef   = useRef(null);
  const overlayRafRef      = useRef(null);
  const lastCheckinRef     = useRef({ employeeId: null, timestamp: 0 });

  const cameraReady = !isJetson && cameraState === "ready";
  const copy = useMemo(() => getGuestResultCopy(result), [result]);

  // ── YOLO ONNX Hook — only active in webcam mode ──────────────────────────
  const {
    modelState,
    modelProgress,
    lastResult: yoloResult,
    getTracksSnapshot,
    getPerfSnapshot,
  } = useYoloDetection({
    videoRef,
    enabled: cameraReady,
    cameraReady,
  });

  // ── Jetson recognition hook ───────────────────────────────────────────────
  const { jetsonResult, jetsonStatus } = useJetsonRecognition({ enabled: isJetson });

  // ── Performance HUD ───────────────────────────────────────────────────────
  const [showPerfHud, setShowPerfHud] = useState(false);
  useEffect(() => {
    function onKey(e) {
      if (e.key === "p" || e.key === "P") {
        if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
        setShowPerfHud((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Merge YOLO results (webcam mode) ─────────────────────────────────────
  useEffect(() => {
    if (!yoloResult) return;
    const payload = yoloResult;
    setResult(payload);

    if (payload?.status === "recognized") {
      const empId = payload?.employee_id;
      const now   = Date.now();
      if (
        empId !== lastCheckinRef.current.employeeId ||
        now - lastCheckinRef.current.timestamp > CHECKIN_COOLDOWN_MS
      ) {
        pushHistory(payload);
        lastCheckinRef.current = { employeeId: empId, timestamp: now };
      }
    }
  }, [yoloResult]);

  // ── Merge Jetson results — update statusText only, history on new check-in ─
  useEffect(() => {
    if (!jetsonResult) return;

    // Always update the "latest result" so confidence ring & cards refresh
    setResult(jetsonResult);

    if (jetsonResult.status === "recognized") {
      const empId = jetsonResult.employee_id;
      const now   = Date.now();
      if (
        empId !== lastCheckinRef.current.employeeId ||
        now - lastCheckinRef.current.timestamp > CHECKIN_COOLDOWN_MS
      ) {
        pushHistory(jetsonResult);
        lastCheckinRef.current = { employeeId: empId, timestamp: now };
      }
    }
  }, [jetsonResult]);

  // ── Bounding box overlay (webcam only) ───────────────────────────────────
  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const vw = video.videoWidth  || 640;
    const vh = video.videoHeight || 480;
    const rw = video.offsetWidth  || vw;
    const rh = video.offsetHeight || vh;

    canvas.width  = rw;
    canvas.height = rh;
    const scaleX = rw / vw;
    const scaleY = rh / vh;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, rw, rh);

    if (modelState !== "ready") return;

    const tracks = getTracksSnapshot();
    for (const track of tracks) {
      const { box, state, result: trackResult } = track;
      if (!box) continue;

      const color = BOX_COLORS[state] || BOX_COLORS.detecting;
      const w  = (box.x2 - box.x1) * scaleX;
      const h  = (box.y2 - box.y1) * scaleY;
      const x1 = rw - box.x2 * scaleX;
      const y1 = box.y1 * scaleY;

      ctx.strokeStyle = color;
      ctx.lineWidth   = 4;
      ctx.strokeRect(x1, y1, w, h);

      const label =
        (state === "recognized" || state === "recognizing") && trackResult?.full_name
          ? trackResult.full_name
          : state === "recognizing"
          ? "Đang xác nhận..."
          : "";

      if (label) {
        ctx.font = "bold 16px system-ui, sans-serif";
        const tw   = ctx.measureText(label).width;
        const padX = 10;
        const labelH = 28;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 + h, tw + padX * 2, labelH);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 + padX, y1 + h + 20);
      }
    }
  }, [modelState, getTracksSnapshot, videoRef]);

  useEffect(() => {
    if (!cameraReady || modelState !== "ready") return;
    const tick = () => {
      drawOverlay();
      overlayRafRef.current = requestAnimationFrame(tick);
    };
    overlayRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (overlayRafRef.current) cancelAnimationFrame(overlayRafRef.current);
    };
  }, [cameraReady, modelState, drawOverlay]);

  // ── Dừng webcam khi chuyển sang Jetson, khởi động lại khi quay về webcam ──
  // Ngắt webcam ngay khi chuyển sang Jetson, không chờ effect chạy lại
  useEffect(() => {
    if (isJetson) {
      stopCamera();
    }
  }, [isJetson, stopCamera]);

  useEffect(() => {
    if (!isJetson) {
      void retryCamera();
    }
  }, [isJetson, retryCamera]);

  // Dọn dẹp khi unmount
  useEffect(() => () => stopCamera(), [stopCamera]);

  // ── Status text (Jetson mode) ─────────────────────────────────────────────
  // Tách riêng khỏi webcam effect để tránh re-run không cần thiết.
  // Không phụ thuộc vào `result` — chỉ dùng jetsonStatus và jetsonResult.
  useEffect(() => {
    if (!isJetson) return;
    if (jetsonStatus === "disconnected") {
      setStatusText("Đang tìm kiếm Jetson camera, vui lòng đợi...");
      return;
    }
    if (jetsonStatus === "scanning") {
      setStatusText("Đang gửi ảnh Jetson lên AI nhận diện...");
      return;
    }
    if (jetsonResult?.status === "recognized" && jetsonResult?.full_name) {
      setStatusText(`Nhận diện: ${jetsonResult.full_name} — Điểm danh thành công.`);
      return;
    }
    if (jetsonResult?.status === "already_checked_in") {
      setStatusText(`${jetsonResult.full_name || "Nhân viên"} đã điểm danh trước đó hôm nay.`);
      return;
    }
    if (jetsonResult?.status === "unknown") {
      setStatusText("Không xác định được khuôn mặt từ Jetson. Đang chờ khung hình tiếp theo...");
      return;
    }
    if (jetsonResult?.status === "no_face") {
      setStatusText("Chưa phát hiện khuôn mặt trong khung hình Jetson.");
      return;
    }
    // Chỉ hiển thị "Jetson sẵn sàng" nếu đã từng kết nối thành công hoặc có kết quả
    if (jetsonStatus === "idle" && (jetsonResult || jetsonResult === null && everConnectedRef?.current)) {
      setStatusText("Jetson sẵn sàng");
      return;
    }
    setStatusText("Jetson camera đang hoạt động");
  }, [isJetson, jetsonStatus, jetsonResult]);

  // ── Status text (Webcam mode) ─────────────────────────────────────────────
  // Tách riêng khỏi Jetson effect. Chỉ chạy khi không ở Jetson mode.
  useEffect(() => {
    if (isJetson) return;
    if (!cameraReady) {
      setStatusText(cameraError || "Camera đang ngoại tuyến. Hãy kiểm tra quyền truy cập hoặc thiết bị.");
      return;
    }
    if (result?.message) {
      setStatusText(result.message);
      return;
    }
    setStatusText("AI đang quét khuôn mặt theo thời gian thực.");
  }, [isJetson, cameraReady, cameraError, result]);

  // ── Helpers ───────────────────────────────────────────────────────────────
  function pushHistory(payload) {
    const confidence = getConfidenceValue(payload?.distance);
    const entry = {
      id:           `${Date.now()}-${Math.random()}`,
      status:       payload?.status || "unknown",
      full_name:    payload?.full_name || "Người lạ / chưa xác định",
      checked_in_at: payload?.checked_in_at || new Date().toISOString(),
      confidence,
    };
    setHistory((current) => [entry, ...current].slice(0, MAX_HISTORY_ITEMS));
  }

  async function handleManualSubmit(event) {
    event.preventDefault();
    if (!manualFile || submissionState === "loading") return;
    setSubmissionState("loading");
    try {
      const queuedPayload = await submitGuestCheckinKpts(manualFile, null);
      const payload =
        queuedPayload?.status === "queued" && queuedPayload?.task_id
          ? await waitGuestCheckinTaskResult(queuedPayload.task_id)
          : queuedPayload;
      setResult(payload);
      if (payload?.status === "recognized") pushHistory(payload);
    } catch (error) {
      setResult({
        status: "network_error",
        message: getFriendlyBackendErrorMessage(error, "Không thể gửi ảnh thủ công đến backend."),
        checked_in_at: new Date().toISOString(),
      });
    } finally {
      setSubmissionState("idle");
    }
  }

  async function handleCameraChange(event) {
    const nextDeviceId = event.target.value;
    if (!nextDeviceId || !selectCamera) return;
    await selectCamera(nextDeviceId);
  }

  const confidence       = getConfidenceValue(result?.distance);
  const confidenceStroke = 339.292;
  const confidenceOffset = confidenceStroke - (confidence / 100) * confidenceStroke;
  const recentPersonName = result?.full_name || "Đang chờ AI xác nhận";

  // ── Jetson display status pill ────────────────────────────────────────────
  // "disconnected" hiển thị neutral (không đỏ) vì user chưa làm gì sai
  const jetsonLiveTone =
    jetsonStatus === "disconnected" || jetsonStatus === "scanning"
      ? "scanning"
      : jetsonResult?.status === "recognized" || jetsonResult?.status === "already_checked_in"
      ? "success"
      : jetsonStatus === "error"
      ? "danger"
      : "scanning";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <main className="kiosk-shell page-transition">
      <section className="kiosk-topbar">
        <div className="stack-sm">
          <span className="section-label">Trạm quét Guardian AI</span>
          <h1>Điểm danh khuôn mặt thông minh</h1>
          <p className="text-secondary">
            Hệ thống nhận diện khuôn mặt với camera thời gian thực, lớp phủ AI và nhật ký cập nhật liên tục.
          </p>
        </div>

        <div className="kiosk-actions">
          {/* ── Camera source toggle ── */}
          <div className="tab-switch" style={{ minHeight: "unset" }}>
            <button
              type="button"
              className={cameraSource === "webcam" ? "active" : ""}
              onClick={() => setCameraSource("webcam")}
            >
              📷 Webcam
            </button>
            <button
              type="button"
              className={cameraSource === "jetson" ? "active" : ""}
              onClick={() => setCameraSource("jetson")}
            >
              🎥 Jetson Camera
            </button>
          </div>

          <Link className="btn btn-secondary" to="/manager/login">
            Mở khu quản trị
          </Link>

          {/* {isJetson ? (
            <span className={`kiosk-live-pill tone-${jetsonLiveTone}`}>
              {jetsonStatus === "disconnected"
                ? "Chưa kết nối"
                : jetsonStatus === "scanning"
                ? "Đang nhận diện"
                : "Jetson Live"}
            </span>
          ) : (
            <span className={`kiosk-live-pill tone-${getTone(cameraReady ? result?.status : "network_error")}`}>
              {getStatusLabel(cameraState)}
            </span>
          )} */}
        </div>
      </section>

      <section className="kiosk-grid">
        {/* ── Video panel ─────────────────────────────────────────────────── */}
        <div className="kiosk-camera-panel panel-dark">
          <div className="kiosk-camera-stage">

            {/* Webcam mode */}
            {!isJetson && (
              <>
                <video
                  ref={videoRef}
                  className="kiosk-video kiosk-video--mirrored"
                  autoPlay
                  playsInline
                  muted
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="kiosk-detection-canvas"
                  style={{
                    position: "absolute",
                    top: 0, left: 0,
                    width: "100%", height: "100%",
                    pointerEvents: "none",
                  }}
                />
                {modelState === "loading" && (
                  <div className="overlay-message" style={{ zIndex: 20 }}>
                    <strong>Đang nạp AI Nhận Diện...</strong>
                    <div style={{ width: "80%", height: 6, background: "rgba(255,255,255,0.15)", borderRadius: 3, margin: "12px auto" }}>
                      <div style={{ width: `${modelProgress}%`, height: "100%", background: "#00e5ff", borderRadius: 3, transition: "width 0.3s" }} />
                    </div>
                    <p style={{ fontSize: "0.85rem", opacity: 0.7 }}>{modelProgress}% — Tải model YOLOv12 (&gt;10MB)</p>
                  </div>
                )}
                {modelState === "error" && (
                  <div className="overlay-message" style={{ zIndex: 20 }}>
                    <strong>Lỗi nạp AI</strong>
                    <p>Không tải được model ONNX.</p>
                  </div>
                )}
                {!cameraReady && (
                  <div className="kiosk-overlay is-error">
                    <div className="overlay-status">
                      <span className="scan-dot" />
                      {getStatusLabel(cameraState)}
                    </div>
                    <div className="overlay-message">
                      <strong>Lỗi camera</strong>
                      <p>{cameraError || "Không kết nối được camera."}</p>
                      <button type="button" className="btn btn-secondary btn-sm" onClick={retryCamera}>
                        Thử lại camera
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Jetson mode — MJPEG stream via <img> */}
            {isJetson && (
              <>
                <JetsonStream streamUrl={JETSON_STREAM_URL} />
                {/* Scanning pulse overlay */}
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    pointerEvents: "none",
                    background:
                      jetsonStatus === "scanning"
                        ? "rgba(0, 229, 255, 0.06)"
                        : "transparent",
                    transition: "background 0.3s",
                  }}
                />
                {/* Status badge top-right */}
                {/* <div className="overlay-status">
                  <span className={`scan-dot${jetsonStatus === "scanning" ? " active" : ""}`} />
                  {jetsonStatus === "disconnected"
                    ? "Chưa kết nối"
                    : jetsonStatus === "scanning"
                    ? "Đang phân tích..."
                    : "Jetson Camera"}
                </div> */}
                {/* Recognition result overlay bottom-left — shows name when recognized */}
                {(result?.status === "recognized" || result?.status === "already_checked_in") &&
                  result?.full_name && (
                    <div
                      className="overlay-message"
                      style={{
                        borderColor:
                          result.status === "recognized"
                            ? "rgba(0,255,128,0.4)"
                            : "rgba(99,179,237,0.35)",
                        background:
                          result.status === "recognized"
                            ? "rgba(0,40,20,0.78)"
                            : "rgba(8,30,55,0.78)",
                      }}
                    >
                      <strong style={{ color: result.status === "recognized" ? "#6ee7b7" : "#93c5fd" }}>
                        {result.status === "recognized" ? "✓ Điểm danh thành công" : "ℹ Đã điểm danh"}
                      </strong>
                      <p style={{ color: "rgba(239,244,255,0.9)", fontWeight: 600, fontSize: "1rem" }}>
                        {result.full_name}
                      </p>
                      {result.employee_code && (
                        <p style={{ color: "rgba(239,244,255,0.6)", fontSize: "0.8rem" }}>
                          {result.employee_code}
                        </p>
                      )}
                    </div>
                  )}
                {result?.status === "no_face" && (
                  <div className="overlay-message">
                    <strong>Chưa thấy khuôn mặt</strong>
                    <p>Đưa mặt vào trung tâm khung hình.</p>
                  </div>
                )}
                {result?.status === "unknown" && (
                  <div className="overlay-message">
                    <strong>Không nhận diện được</strong>
                    <p>Khuôn mặt chưa có trong hệ thống.</p>
                  </div>
                )}
              </>
            )}

            {!isJetson && showPerfHud && <PerfHud getPerfSnapshot={getPerfSnapshot} />}
            {!isJetson && (
              <button
                type="button"
                className="perf-hud-toggle"
                onClick={() => setShowPerfHud((v) => !v)}
                title="Bật/tắt Performance HUD (phím P)"
              >
                ⚡
              </button>
            )}
          </div>

          <div className="kiosk-toolbar">
            <div className="stack-sm">
              <span className="section-label">Điều khiển quét</span>
              <strong>
                {isJetson
                  ? `Jetson Camera`
                  : "Camera đang quét liên tục"}
              </strong>
            </div>

            <div className="kiosk-toolbar-actions">
              {/* Webcam device picker — only in webcam mode */}
              {!isJetson && cameraDevices.length > 0 && (
                <label className="kiosk-camera-select" htmlFor="camera-device-select">
                  <span className="text-muted">Nguồn camera</span>
                  <select
                    id="camera-device-select"
                    value={selectedCameraId || cameraDevices[0].deviceId}
                    onChange={handleCameraChange}
                    disabled={submissionState === "loading"}
                  >
                    {cameraDevices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {/* Jetson connection info */}
              {isJetson && (
                <span className="pill" style={{ fontSize: 12 }}>
                  {JETSON_STREAM_URL}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* ── Side panel — unchanged layout ─────────────────────────────── */}
        <aside className="kiosk-side-panel">
          <div className="glass-panel kiosk-result-card">
            <div className="row-between">
              <div className="stack-sm">
                <span className="section-label">Kết quả AI</span>
                <h2>Người vừa quét</h2>
              </div>
              <span
                className={`badge badge-${
                  getTone(result?.status) === "success"
                    ? "success"
                    : getTone(result?.status) === "warning"
                    ? "warning"
                    : getTone(result?.status) === "danger"
                    ? "error"
                    : "info"
                }`}
              >
                {copy?.label || "Đang quét"}
              </span>
            </div>

            <div className="kiosk-profile">
              <div className="kiosk-avatar">{getEmployeeInitials(recentPersonName)}</div>
              <div className="stack-sm">
                <strong>{recentPersonName}</strong>
                <span className="text-secondary">
                  {result?.employee_code || "Luồng khách Guardian AI"}
                </span>
                <span className="text-muted">{formatDateTime(result?.checked_in_at)}</span>
              </div>
            </div>

            <div className="kiosk-confidence">
              <div className="confidence-ring">
                <svg viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="54" />
                  <circle
                    className="progress-ring"
                    cx="60"
                    cy="60"
                    r="54"
                    style={{
                      strokeDasharray: confidenceStroke,
                      strokeDashoffset: confidenceOffset,
                    }}
                  />
                </svg>
                <div>
                  <strong>{confidence.toFixed(1)}%</strong>
                  <span>Khớp</span>
                </div>
              </div>

              <div className="stack-sm">
                <div className="pill">
                  {isJetson ? "Jetson trực tuyến" : cameraReady ? "AI trực tuyến" : "Camera ngoại tuyến"}
                </div>
                {/* ── The only output area updated by Jetson recognition ── */}
                <p className="text-secondary">{statusText}</p>
              </div>
            </div>

            <div className="kiosk-meta-grid">
              <div className="kiosk-meta">
                <span>Trạng thái</span>
                <strong>
                  {isJetson
                    ? jetsonStatus === "disconnected"
                      ? "Chưa kết nối"
                      : jetsonStatus === "scanning"
                      ? "Đang phân tích"
                      : "Jetson sẵn sàng"
                    : getStatusLabel(cameraState)}
                </strong>
              </div>
              <div className="kiosk-meta">
                <span>Điểm danh</span>
                <strong>{formatTime(result?.checked_in_at)}</strong>
              </div>
              <div className="kiosk-meta">
                <span>Nguồn</span>
                <strong>{isJetson ? "Jetson Camera" : cameraState}</strong>
              </div>
              <div className="kiosk-meta">
                <span>Ghi chú AI</span>
                <strong>{copy?.message || "Đang chờ dữ liệu mới"}</strong>
              </div>
            </div>
          </div>

          <div className="glass-panel kiosk-history-card">
            <div className="row-between">
              <div className="stack-sm">
                <span className="section-label">Lượt quét gần đây</span>
                <h2>Lịch sử gần nhất</h2>
              </div>
              <span className="pill">{history.length} bản ghi</span>
            </div>

            <div className="kiosk-history-list">
              {history.length === 0 ? (
                <div className="empty-state">
                  <h3>Chưa có log</h3>
                  <p>AI sẽ cập nhật danh sách này khi có người điểm danh thành công.</p>
                </div>
              ) : (
                history.map((entry) => (
                  <div key={entry.id} className="kiosk-history-item">
                    <div className="kiosk-history-avatar">{getEmployeeInitials(entry.full_name)}</div>
                    <div className="stack-sm kiosk-history-copy">
                      <strong>{entry.full_name}</strong>
                      <span className="text-secondary">{formatDateTime(entry.checked_in_at)}</span>
                    </div>
                    <div className="stack-sm kiosk-history-side">
                      <span className={`badge ${getHistoryBadge(entry)}`}>{entry.status}</span>
                      <strong>{entry.confidence.toFixed(1)}%</strong>
                    </div>
                  </div>
                ))
              )}
            </div>

            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setShowFallback((current) => !current)}
            >
              {showFallback ? "Đóng tải ảnh thủ công" : "Camera lỗi? Tải ảnh thủ công"}
            </button>

            {showFallback && (
              <form className="kiosk-upload-panel" onSubmit={handleManualSubmit}>
                <div className="field">
                  <label htmlFor="manual-upload">Ảnh khuôn mặt</label>
                  <input
                    id="manual-upload"
                    type="file"
                    accept="image/*"
                    onChange={(event) => setManualFile(event.target.files?.[0] ?? null)}
                  />
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!manualFile || submissionState === "loading"}
                >
                  Gửi ảnh lên AI
                </button>
              </form>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}