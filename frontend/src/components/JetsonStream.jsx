import React, { useCallback, useEffect, useRef } from "react";

export const JETSON_STREAM_URL = "/jetson/stream";

// ── JetsonStream ─────────────────────────────────────────────────────────────
// Tách thành component riêng + React.memo để MJPEG <img> không bị remount
// mỗi khi parent re-render. Khi Jetson không kết nối được, browser sẽ retry
// liên tục gây broken-image flicker — onError dùng exponential backoff để
// giảm tần suất retry thay vì để browser tự retry ngay lập tức.
const JetsonStream = React.memo(function JetsonStream({ streamUrl, onFrame }) {
  const url = streamUrl || JETSON_STREAM_URL;
  const imgRef = useRef(null);
  const retryTimer = useRef(null);
  const retryDelay = useRef(2000);

  const scheduleRetry = useCallback(() => {
    if (retryTimer.current) return;
    retryTimer.current = setTimeout(() => {
      retryTimer.current = null;
      if (imgRef.current) {
        imgRef.current.src = `${url}?t=${Date.now()}`;
      }
      retryDelay.current = Math.min(retryDelay.current * 2, 30000);
    }, retryDelay.current);
  }, [url]);

  const handleLoad = useCallback(() => {
    retryDelay.current = 2000;
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    console.debug("[JetsonStream] image loaded", {
      url,
      naturalWidth: imgRef.current?.naturalWidth,
      naturalHeight: imgRef.current?.naturalHeight,
    });
    if (onFrame && imgRef.current) {
      try { onFrame(imgRef.current) } catch (e) { /* swallow */ }
    }
  }, [onFrame]);

  useEffect(() => {
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, []);

  return (
    <img
      ref={imgRef}
      src={url}
      alt="Jetson camera stream"
      className="kiosk-video"
      style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
      onLoad={handleLoad}
      onError={scheduleRetry}
    />
  );
});

export default JetsonStream;
