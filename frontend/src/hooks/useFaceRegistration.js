import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, enrollEmployeeFacesBatch, getFaceSamples } from "../lib/api";
import { getFriendlyBackendErrorMessage } from "../lib/errorMessages";
import {
  buildCaptureFeedback,
  CAPTURE_LOOP_INTERVAL_MS,
  CAPTURE_SOFT_WARNING_MS,
  computeBlurScore,
  DEFAULT_CAPTURE_CONFIG,
  evaluateFrameQuality,
  mirrorHintPose,
  normalizeCaptureConfig,
  shouldExtendCaptureAfterBatchError,
} from "../lib/faceRegistrationCapture";
import { cropFace, detectFaces, isModelLoaded, loadModel } from "../ai";

const MOCK_EMPLOYEE = {
  department: "Phòng Kế toán",
  employee_code: "NV001",
  full_name: "Nguyễn Văn A",
  id: "NV001",
  is_active: true,
};

export function useFaceRegistration(employeeId, { onUnauthenticated } = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const workCanvasRef = useRef(null);
  const captureTimeoutRef = useRef(null);
  const sessionTokenRef = useRef(0);
  const sessionStartAtRef = useRef(0);
  const lastAcceptedAtRef = useRef(Number.NaN);
  const acceptedFramesRef = useRef([]);
  const captureConfigRef = useRef(normalizeCaptureConfig(DEFAULT_CAPTURE_CONFIG));
  const sessionStatusRef = useRef("idle");
  const mountedRef = useRef(false);
  const runCaptureLoopRef = useRef(null);
  const onAuthRef = useRef(onUnauthenticated);
  onAuthRef.current = onUnauthenticated;

  const [employee, setEmployee] = useState(MOCK_EMPLOYEE);
  const [captureConfig, setCaptureConfig] = useState(captureConfigRef.current);
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [targetCount, setTargetCount] = useState(captureConfigRef.current.minFrames);
  const [acceptedFrames, setAcceptedFrames] = useState([]);
  const [thumbnailFrames, setThumbnailFrames] = useState([]);
  const [liveFeedback, setLiveFeedback] = useState(
    buildCaptureFeedback("idle", { targetCount: captureConfigRef.current.minFrames }),
  );
  const [elapsedMs, setElapsedMs] = useState(0);
  const [softWarningVisible, setSoftWarningVisible] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [saveState, setSaveState] = useState("idle");
  const [cameraState, setCameraState] = useState("initializing");
  const [cameraError, setCameraError] = useState("");
  const [modelState, setModelState] = useState("idle");

  const updateSessionStatus = useCallback((nextStatus) => {
    sessionStatusRef.current = nextStatus;
    setSessionStatus(nextStatus);
  }, []);

  const revokePreviewUrls = useCallback((frames) => {
    frames.forEach((frame) => {
      if (frame?.previewUrl) {
        URL.revokeObjectURL(frame.previewUrl);
      }
    });
  }, []);

  const clearCaptureLoop = useCallback(() => {
    sessionTokenRef.current += 1;
    if (captureTimeoutRef.current) {
      clearTimeout(captureTimeoutRef.current);
      captureTimeoutRef.current = null;
    }
  }, []);

  const resetCaptureState = useCallback(({ keepMessage = false } = {}) => {
    clearCaptureLoop();
    revokePreviewUrls(acceptedFramesRef.current);
    acceptedFramesRef.current = [];
    lastAcceptedAtRef.current = Number.NaN;
    sessionStartAtRef.current = 0;
    setAcceptedFrames([]);
    setThumbnailFrames([]);
    setElapsedMs(0);
    setSoftWarningVisible(false);
    setTargetCount(captureConfigRef.current.minFrames);
    setSaveState("idle");
    if (!keepMessage) {
      setSaveMessage("");
    }
  }, [clearCaptureLoop, revokePreviewUrls]);

  useEffect(() => {
    mountedRef.current = true;

    async function bootstrap() {
      try {
        const payload = await getFaceSamples(employeeId);
        if (!mountedRef.current) return;
        setEmployee(payload?.employee || MOCK_EMPLOYEE);
        const nextConfig = normalizeCaptureConfig(payload?.capture_config);
        captureConfigRef.current = nextConfig;
        setCaptureConfig(nextConfig);
        setTargetCount(nextConfig.minFrames);
        setLiveFeedback(buildCaptureFeedback("idle", { targetCount: nextConfig.minFrames }));
      } catch (error) {
        if (!mountedRef.current) return;
        if (error?.status === 401) {
          onAuthRef.current?.();
          return;
        }
        setEmployee((currentEmployee) => ({ ...currentEmployee, id: employeeId || currentEmployee.id }));
      }

      if (!videoRef.current) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        if (!mountedRef.current || !videoRef.current) return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (!mountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraState("ready");
      } catch (error) {
        if (!mountedRef.current) return;
        setCameraState("error");
        setCameraError(error?.message || "Không thể mở camera.");
        return;
      }

      if (!workCanvasRef.current) {
        workCanvasRef.current = document.createElement("canvas");
      }

      if (isModelLoaded()) {
        if (mountedRef.current) setModelState("ready");
        return;
      }

      if (mountedRef.current) setModelState("loading");
      try {
        await loadModel(() => {});
        if (mountedRef.current) setModelState("ready");
      } catch (error) {
        console.error("[FaceRegistration] YOLO load failed:", error);
        if (mountedRef.current) setModelState("error");
      }
    }

    void bootstrap();

    return () => {
      mountedRef.current = false;
      clearCaptureLoop();
      revokePreviewUrls(acceptedFramesRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [clearCaptureLoop, employeeId, revokePreviewUrls]);

  useEffect(() => {
    if (sessionStatus !== "collecting") {
      setElapsedMs(0);
      setSoftWarningVisible(false);
      return undefined;
    }

    const intervalHandle = setInterval(() => {
      const nextElapsed = Math.max(0, Date.now() - sessionStartAtRef.current);
      setElapsedMs(nextElapsed);
      setSoftWarningVisible(nextElapsed >= CAPTURE_SOFT_WARNING_MS);
    }, 250);

    return () => clearInterval(intervalHandle);
  }, [sessionStatus]);

  const submitBatch = useCallback(async () => {
    const activeTargetCount = targetCount;
    updateSessionStatus("uploading");
    setSaveState("idle");
    setSaveMessage("");
    setLiveFeedback(buildCaptureFeedback("uploading", { acceptedCount: acceptedFramesRef.current.length, targetCount: activeTargetCount }));

    try {
      await enrollEmployeeFacesBatch(employeeId, acceptedFramesRef.current);
      if (!mountedRef.current) return;
      updateSessionStatus("success");
      setSaveState("success");
      setSaveMessage(`Hoàn tất đăng ký thành công ${acceptedFramesRef.current.length} mẫu ảnh!`);
      setLiveFeedback(buildCaptureFeedback("success", { acceptedCount: acceptedFramesRef.current.length, targetCount: activeTargetCount }));
      setEmployee((currentEmployee) => ({ ...currentEmployee, registration_status: "Đã đăng ký" }));
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        onAuthRef.current?.();
        return;
      }
      if (!mountedRef.current) return;

      if (shouldExtendCaptureAfterBatchError(error, {
        acceptedCount: acceptedFramesRef.current.length,
        currentTargetCount: activeTargetCount,
        maxFrames: captureConfigRef.current.maxFrames,
      })) {
        const extendedTargetCount = captureConfigRef.current.maxFrames;
        const nextToken = sessionTokenRef.current + 1;
        sessionTokenRef.current = nextToken;
        setTargetCount(extendedTargetCount);
        updateSessionStatus("collecting");
        setSaveState("idle");
        setSaveMessage("Máy chủ cần thêm vài khung hình khác nhau. Hãy xoay tiếp để hệ thống gom thêm ảnh và gửi lại.");
        setLiveFeedback(buildCaptureFeedback("needs_more_frames", {
          acceptedCount: acceptedFramesRef.current.length,
          targetCount: extendedTargetCount,
        }));
        captureTimeoutRef.current = setTimeout(() => {
          void runCaptureLoopRef.current?.(nextToken);
        }, CAPTURE_LOOP_INTERVAL_MS);
        return;
      }

      updateSessionStatus("error");
      setSaveState("error");
      setSaveMessage(
        getFriendlyBackendErrorMessage(error, "Không thể gửi dữ liệu khuôn mặt lên máy chủ."),
      );
      setLiveFeedback(buildCaptureFeedback("error", { acceptedCount: acceptedFramesRef.current.length, targetCount: activeTargetCount }));
    }
  }, [employeeId, targetCount, updateSessionStatus]);

  const scheduleNextCapture = useCallback((token, runner) => {
    if (!mountedRef.current || token !== sessionTokenRef.current || sessionStatusRef.current !== "collecting") {
      return;
    }

    captureTimeoutRef.current = setTimeout(() => {
      void runner(token);
    }, CAPTURE_LOOP_INTERVAL_MS);
  }, []);

  const runCaptureLoop = useCallback(async (token) => {
    if (!mountedRef.current || token !== sessionTokenRef.current || sessionStatusRef.current !== "collecting") {
      return;
    }

    const videoElement = videoRef.current;
    if (!videoElement || videoElement.readyState < 2 || modelState !== "ready" || !workCanvasRef.current) {
      scheduleNextCapture(token, runCaptureLoop);
      return;
    }

    const nowMs = Date.now();
    let detections = [];

    try {
      const detectResult = await detectFaces(videoElement, workCanvasRef.current);
      detections = detectResult.detections || detectResult;
    } catch (error) {
      console.error("[FaceRegistration] detectFaces failed:", error);
      setLiveFeedback(
        buildCaptureFeedback("low_confidence", {
          acceptedCount: acceptedFramesRef.current.length,
          targetCount,
        }),
      );
      scheduleNextCapture(token, runCaptureLoop);
      return;
    }

    if (!mountedRef.current || token !== sessionTokenRef.current || sessionStatusRef.current !== "collecting") {
      return;
    }

    const precheck = evaluateFrameQuality({
      detections,
      videoWidth: videoElement.videoWidth,
      videoHeight: videoElement.videoHeight,
      nowMs,
      lastAcceptedAtMs: lastAcceptedAtRef.current,
      blurScore: Number.POSITIVE_INFINITY,
      config: captureConfigRef.current,
    });

    if (!precheck.accepted) {
      setLiveFeedback(
        buildCaptureFeedback(precheck.reason, {
          acceptedCount: acceptedFramesRef.current.length,
          targetCount,
        }),
      );
      scheduleNextCapture(token, runCaptureLoop);
      return;
    }

    let cropResult = null;
    try {
      cropResult = await cropFace(videoElement, precheck.detection, 0.4);
    } catch (error) {
      console.error("[FaceRegistration] cropFace failed:", error);
    }

    if (!mountedRef.current || token !== sessionTokenRef.current || sessionStatusRef.current !== "collecting") {
      return;
    }

    if (!cropResult?.blob || !cropResult?.canvas) {
      setLiveFeedback(
        buildCaptureFeedback("blurry", {
          acceptedCount: acceptedFramesRef.current.length,
          targetCount,
        }),
      );
      scheduleNextCapture(token, runCaptureLoop);
      return;
    }

    const cropContext = cropResult.canvas.getContext("2d", { willReadFrequently: true });
    const imageData = cropContext?.getImageData(0, 0, cropResult.canvas.width, cropResult.canvas.height);
    const blurScore = computeBlurScore(imageData);
    const finalDecision = evaluateFrameQuality({
      detections: [precheck.detection],
      videoWidth: videoElement.videoWidth,
      videoHeight: videoElement.videoHeight,
      nowMs,
      lastAcceptedAtMs: lastAcceptedAtRef.current,
      blurScore,
      config: captureConfigRef.current,
    });

    if (!finalDecision.accepted) {
      setLiveFeedback(
        buildCaptureFeedback(finalDecision.reason, {
          acceptedCount: acceptedFramesRef.current.length,
          targetCount,
        }),
      );
      scheduleNextCapture(token, runCaptureLoop);
      return;
    }

    const nextAcceptedFrame = {
      blob: cropResult.blob,
      capturedAtMs: nowMs - sessionStartAtRef.current,
      detectorScore: precheck.detection.score,
      blurScore,
      hintPose: mirrorHintPose(finalDecision.hintPose),
      previewUrl: URL.createObjectURL(cropResult.blob),
    };

    const nextAcceptedFrames = [...acceptedFramesRef.current, nextAcceptedFrame];
    acceptedFramesRef.current = nextAcceptedFrames;
    lastAcceptedAtRef.current = nowMs;
    setAcceptedFrames(nextAcceptedFrames);
    setThumbnailFrames(nextAcceptedFrames.slice(-captureConfigRef.current.thumbnailLimit));
    setLiveFeedback(
      buildCaptureFeedback("accepted", {
        acceptedCount: nextAcceptedFrames.length,
        targetCount,
      }),
    );

    if (nextAcceptedFrames.length >= targetCount) {
      clearCaptureLoop();
      void submitBatch();
      return;
    }

    scheduleNextCapture(token, runCaptureLoop);
  }, [clearCaptureLoop, modelState, scheduleNextCapture, submitBatch, targetCount]);

  runCaptureLoopRef.current = runCaptureLoop;

  const startRecording = useCallback(() => {
    if (sessionStatusRef.current !== "idle") return;
    if (modelState !== "ready" || cameraState !== "ready") return;

    resetCaptureState();
    updateSessionStatus("collecting");
    sessionTokenRef.current += 1;
    sessionStartAtRef.current = Date.now();
    setTargetCount(captureConfigRef.current.minFrames);
    setLiveFeedback(buildCaptureFeedback("no_face", { targetCount: captureConfigRef.current.minFrames }));
    void runCaptureLoop(sessionTokenRef.current);
  }, [cameraState, modelState, resetCaptureState, runCaptureLoop, updateSessionStatus]);

  const resetRegistration = useCallback(() => {
    resetCaptureState();
    updateSessionStatus("idle");
    setLiveFeedback(buildCaptureFeedback("idle", { targetCount: captureConfigRef.current.minFrames }));
  }, [resetCaptureState, updateSessionStatus]);

  const canStart = sessionStatus === "idle" && modelState === "ready" && cameraState === "ready";
  const canReset = sessionStatus === "collecting" || sessionStatus === "error" || acceptedFrames.length > 0;

  return {
    videoRef,
    cameraState,
    cameraError,
    modelState,
    sessionStatus,
    acceptedFrames,
    acceptedCount: acceptedFrames.length,
    targetCount,
    thumbnailFrames,
    thumbnailLimit: captureConfig.thumbnailLimit,
    captureConfig,
    liveFeedback,
    elapsedMs,
    softWarningVisible,
    saveMessage,
    saveState,
    canStart,
    canReset,
    employee: {
      ...employee,
      registration_status:
        employee.registration_status ||
        (sessionStatus === "success" ? "Đã đăng ký" : "Chưa đăng ký"),
    },
    startRecording,
    resetRegistration,
  };
}
