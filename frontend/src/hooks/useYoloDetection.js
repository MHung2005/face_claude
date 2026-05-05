// ============================================================
// useYoloDetection.js — React Hook: YOLO ONNX trên browser
// ============================================================
// Quản lý toàn bộ vòng đời ONNX detection:
//   1. Load model (với progress bar)
//   2. Chạy detection loop qua requestAnimationFrame
//   3. Track khuôn mặt qua các frame (Centroid Tracker)
//   4. Trigger gửi crop+keypoints khi đủ điều kiện
//   5. Vẽ bounding box lên canvas overlay
// ============================================================

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  computeIoU,
  cropFace,
  detectFaces,
  isModelLoaded,
  loadModel,
} from '../ai'
import { submitGuestCheckinKpts, waitGuestCheckinTaskResult } from '../lib/guestApi'
import { getFriendlyBackendErrorMessage } from '../lib/errorMessages'

// ============================================================
// Cấu hình State Machine
// ============================================================
const STABLE_FRAMES_REQUIRED = 2  // Giảm yêu cầu đứng yên lâu trước khi gửi nhận diện
const MIN_FACE_AREA_RATIO = 0.025 // Cho phép mặt nhỏ hơn một chút để bắt nhanh hơn
const MIN_CONFIDENCE = 0.4        // Nới ngưỡng confidence để nhận diện sớm hơn
const COOLDOWN_MS = 1500          // Giảm thời gian chờ giữa các lần nhận diện
const MAX_RETRIES = 3             // Số lần thử lại cho unknown
const TRACK_LOST_FRAMES = 5       // Xóa ghost track nhanh (0.5s), tránh bbox chồng chéo
const DETECTION_INTERVAL_MS = 100 // Chạy detect mỗi 100ms (~10fps YOLO)
const DEBUG_DETECTION = false

/**
 * Tính khoảng cách Euclidean giữa 2 tâm box.
 */
function centroidDistance(boxA, boxB) {
  const cxA = (boxA.x1 + boxA.x2) / 2
  const cyA = (boxA.y1 + boxA.y2) / 2
  const cxB = (boxB.x1 + boxB.x2) / 2
  const cyB = (boxB.y1 + boxB.y2) / 2
  return Math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
}

/**
 * Custom React Hook: Quản lý YOLO ONNX detection trên browser.
 *
 * @param {object} options
 * @param {React.RefObject<HTMLVideoElement>} options.videoRef
 * @param {boolean} options.enabled - Bật/tắt detection loop
 * @param {boolean} options.cameraReady - Camera đã sẵn sàng chưa
 * @returns {{ modelState, modelProgress, detections, tracks, onResult }}
 */
export function useYoloDetection({ videoRef, enabled, cameraReady }) {
  // --- State public ---
  const [modelState, setModelState] = useState('idle')  // idle | loading | ready | error
  const [modelProgress, setModelProgress] = useState(0)
  const [detections, setDetections] = useState([])       // Mảng detection hiện tại
  const [lastResult, setLastResult] = useState(null)      // Kết quả nhận diện mới nhất

  // --- Refs nội bộ ---
  const workCanvasRef = useRef(null)
  const tracksRef = useRef(new Map())     // Map<trackId, trackState>
  const nextTrackIdRef = useRef(1)
  const loopActiveRef = useRef(false)
  const lastDetectTimeRef = useRef(0)
  const perfRef = useRef({ detect: null, crop: null, network: null, backend: null })

  // ============================================================
  // Bước 1: Load model ONNX khi component mount
  // ============================================================
  useEffect(() => {
    if (isModelLoaded()) {
      setModelState('ready')
      setModelProgress(100)
      return
    }

    let cancelled = false
    setModelState('loading')

    loadModel((percent) => {
      if (!cancelled) setModelProgress(percent)
    })
      .then(() => {
        if (!cancelled) {
          setModelState('ready')
          setModelProgress(100)
        }
      })
      .catch((err) => {
        console.error('[useYoloDetection] Model load failed:', err)
        if (!cancelled) setModelState('error')
      })

    return () => { cancelled = true }
  }, [])

  // ============================================================
  // Bước 2: Tạo work canvas (dùng để preprocessing)
  // ============================================================
  useEffect(() => {
    if (!workCanvasRef.current) {
      workCanvasRef.current = document.createElement('canvas')
    }
  }, [])

  // ============================================================
  // Bước 3: Xử lý kết quả nhận diện (callback từ API)
  // ============================================================
  const handleRecognitionResult = useCallback((trackId, payload) => {
    const tracks = tracksRef.current
    const track = tracks.get(trackId)
    if (!track) {
      // Task async có thể hoàn tất sau khi track đã mất do người dùng di chuyển.
      // Vẫn phát kết quả để UI cập nhật check-in thay vì bỏ qua.
      setLastResult({ ...payload, trackId })
      return
    }

    if (payload.status === 'recognized' || payload.status === 'already_checked_in') {
      track.state = 'recognized'
      track.result = payload
    } else if (payload.status === 'unknown') {
      track.retries += 1
      if (track.retries >= MAX_RETRIES) {
        track.state = 'unknown'
      } else {
        track.state = 'detecting'
        track.stableFrames = 0
      }
    } else {
      // no_face, error, etc. → reset
      track.state = 'detecting'
      track.stableFrames = 0
    }

    track.inflight = false
    setLastResult({ ...payload, trackId })
  }, [])

  // ============================================================
  // Bước 4: Gửi crop+keypoints lên backend (trigger khi đủ điều kiện)
  // ============================================================
  const sendCropToBackend = useCallback(async (trackId, videoEl, detection) => {
    const tracks = tracksRef.current
    const track = tracks.get(trackId)
    if (!track || track.inflight) return

    track.inflight = true
    track.state = 'recognizing'

    try {
      const tCrop0 = performance.now()
      const cropResult = await cropFace(videoEl, detection, 0.4)
      const tCrop1 = performance.now()
      perfRef.current.crop = Math.round((tCrop1 - tCrop0) * 10) / 10

      if (!cropResult) {
        track.inflight = false
        track.state = 'detecting'
        return
      }

      const tNet0 = performance.now()
      const payload = await submitGuestCheckinKpts(
        cropResult.blob,
        cropResult.localKeypoints,
      )
      let finalPayload = payload
      if (payload?.status === 'queued' && payload?.task_id) {
        finalPayload = await waitGuestCheckinTaskResult(payload.task_id)
      }
      const tNet1 = performance.now()
      perfRef.current.network = Math.round((tNet1 - tNet0) * 10) / 10
      perfRef.current.backend = finalPayload?._timing || null

      handleRecognitionResult(trackId, finalPayload)
    } catch (error) {
      handleRecognitionResult(trackId, {
        status: 'network_error',
        message: getFriendlyBackendErrorMessage(error, 'Lỗi gửi dữ liệu.'),
        checked_in_at: new Date().toISOString(),
      })
    }
  }, [handleRecognitionResult])

  // ============================================================
  // Bước 5: Centroid Tracker — gán ID cho khuôn mặt qua các frame
  // ============================================================
  const updateTracks = useCallback((newDetections, videoEl) => {
    const tracks = tracksRef.current
    const vw = videoEl.videoWidth || 1
    const vh = videoEl.videoHeight || 1
    const frameArea = vw * vh

    // Đánh dấu tất cả tracks là "chưa match"
    for (const track of tracks.values()) {
      track.matched = false
    }

    // Helper: kiểm tra nếu detection mới "trông giống người khác" so với track cũ
    // Dùng khi track đã recognized — phát hiện swap người mà không có gap frame
    const looksLikeDifferentPerson = (track, newBox) => {
      if (track.state !== 'recognized') return false
      const dist = centroidDistance(newBox, track.lastBox)
      // Tâm bbox nhảy >8% chiều rộng video → quá xa cho cùng 1 người trong 100ms
      if (dist > vw * 0.08) return true
      // Kích thước bbox thay đổi >60% → mặt mới to/nhỏ hơn hẳn
      const oldArea = (track.lastBox.x2 - track.lastBox.x1) * (track.lastBox.y2 - track.lastBox.y1)
      const newArea = (newBox.x2 - newBox.x1) * (newBox.y2 - newBox.y1)
      if (oldArea > 0 && Math.abs(newArea - oldArea) / oldArea > 0.6) return true
      return false
    }

    const resetTrackIdentity = (track) => {
      track.state = 'detecting'
      track.result = null
      track.retries = 0
      track.inflight = false
      track.stableFrames = 0
    }

    // Gán detection → track gần nhất (Centroid matching)
    for (const det of newDetections) {
      let bestTrackId = null
      let bestDist = Infinity

      for (const [id, track] of tracks.entries()) {
        if (track.matched) continue
        const dist = centroidDistance(det.box, track.lastBox)
        if (dist < bestDist) {
          bestDist = dist
          bestTrackId = id
        }
      }

      // Ngưỡng: nếu centroid quá xa (>25% chiều rộng video) → kiểm tra thêm IoU
      const maxDist = vw * 0.25
      if (bestTrackId !== null && bestDist < maxDist) {
        const track = tracks.get(bestTrackId)
        // Reset nếu track mất match trước đó HOẶC trông giống người khác
        if ((track.missedFrames > 0 && track.state === 'recognized') ||
            looksLikeDifferentPerson(track, det.box)) {
          resetTrackIdentity(track)
        }
        track.matched = true
        track.lastBox = det.box
        track.lastDetection = det
        track.missedFrames = 0
        track.stableFrames += 1
      } else {
        // Fallback: kiểm tra IoU trước khi tạo track mới
        let iouMatchId = null
        for (const [id, track] of tracks.entries()) {
          if (track.matched) continue
          if (computeIoU(det.box, track.lastBox) > 0.3) {
            iouMatchId = id
            break
          }
        }

        if (iouMatchId !== null) {
          const track = tracks.get(iouMatchId)
          if ((track.missedFrames > 0 && track.state === 'recognized') ||
              looksLikeDifferentPerson(track, det.box)) {
            resetTrackIdentity(track)
          }
          track.matched = true
          track.lastBox = det.box
          track.lastDetection = det
          track.missedFrames = 0
          track.stableFrames += 1
        } else {
          // Thực sự là mặt mới → tạo track mới
          const newId = nextTrackIdRef.current++
          tracks.set(newId, {
            lastBox: det.box,
            lastDetection: det,
            state: 'detecting',
            stableFrames: 1,
            missedFrames: 0,
            retries: 0,
            inflight: false,
            result: null,
            cooldownUntil: 0,
            matched: true,
          })
        }
      }
    }

    // Tăng missedFrames cho track không match, xóa track mất quá lâu
    for (const [id, track] of tracks.entries()) {
      if (!track.matched) {
        track.missedFrames += 1
        if (track.missedFrames > TRACK_LOST_FRAMES) {
          tracks.delete(id)
        }
      }
    }

    // Check trigger conditions cho từng track
    for (const [id, track] of tracks.entries()) {
      if (track.state !== 'detecting') continue
      if (track.inflight) continue
      if (Date.now() < track.cooldownUntil) continue

      const det = track.lastDetection
      if (!det) continue

      // Kiểm tra diện tích
      const bw = det.box.x2 - det.box.x1
      const bh = det.box.y2 - det.box.y1
      const faceArea = bw * bh
      if (faceArea / frameArea < MIN_FACE_AREA_RATIO) {
        if (DEBUG_DETECTION) {
          console.warn(`[Track#${id}] SKIP: faceArea ${(faceArea/frameArea).toFixed(4)} < min ${MIN_FACE_AREA_RATIO}`)
        }
        continue
      }

      // Kiểm tra confidence
      if (det.score < MIN_CONFIDENCE) {
        if (DEBUG_DETECTION) {
          console.warn(`[Track#${id}] SKIP: score ${det.score.toFixed(3)} < min ${MIN_CONFIDENCE}`)
        }
        continue
      }

      // Kiểm tra stable frames
      if (track.stableFrames < STABLE_FRAMES_REQUIRED) {
        if (DEBUG_DETECTION) {
          console.warn(`[Track#${id}] SKIP: stableFrames ${track.stableFrames} < required ${STABLE_FRAMES_REQUIRED}`)
        }
        continue
      }

      if (DEBUG_DETECTION) {
        console.warn(`[Track#${id}] TRIGGER → backend (score=${det.score.toFixed(3)}, areaRatio=${(faceArea/frameArea).toFixed(4)})`)
      }
      // ĐỦ ĐIỀU KIỆN → Gửi crop!
      track.cooldownUntil = Date.now() + COOLDOWN_MS
      sendCropToBackend(id, videoEl, det)
    }
  }, [sendCropToBackend])

  // ============================================================
  // Bước 6: Detection Loop (chạy liên tục khi enabled)
  // ============================================================
  useEffect(() => {
    if (modelState !== 'ready' || !enabled || !cameraReady) {
      loopActiveRef.current = false
      return
    }

    loopActiveRef.current = true
    let rafId = null

    const loop = async () => {
      if (!loopActiveRef.current) return

      const now = Date.now()
      if (now - lastDetectTimeRef.current >= DETECTION_INTERVAL_MS) {
        lastDetectTimeRef.current = now

        const videoEl = videoRef.current
        if (videoEl && videoEl.readyState >= 2 && workCanvasRef.current) {
          try {
            const { detections: dets, timing: detectTiming } = await detectFaces(videoEl, workCanvasRef.current)
            if (detectTiming) perfRef.current.detect = detectTiming
            if (DEBUG_DETECTION && dets.length > 0) {
              console.warn(`[detectFaces] ✓ ${dets.length} face(s) detected`)
            }
            setDetections(dets)
            updateTracks(dets, videoEl)
          } catch (err) {
            console.error('[useYoloDetection] Detection error:', err)
          }
        }
      }

      rafId = requestAnimationFrame(loop)
    }

    rafId = requestAnimationFrame(loop)

    return () => {
      loopActiveRef.current = false
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [modelState, enabled, cameraReady, videoRef, updateTracks])

  // ============================================================
  // Bước 7: Export trạng thái tracks để UI vẽ bounding box
  // ============================================================
  const getTracksSnapshot = useCallback(() => {
    const result = []
    for (const [id, track] of tracksRef.current.entries()) {
      // Ẩn ghost track đã mất match — tránh bbox chồng chéo
      if (track.missedFrames > 0) continue
      result.push({
        id,
        box: track.lastBox,
        state: track.state,
        result: track.result,
        score: track.lastDetection?.score || 0,
      })
    }
    if (result.length > 0) {
      if (!DEBUG_DETECTION) {
        return result
      }
      console.warn(`[tracks] ${result.length} track(s):`, result.map(t => `#${t.id} state=${t.state} box=(${t.box?.x1?.toFixed(0)},${t.box?.y1?.toFixed(0)}) score=${t.score.toFixed(3)}`))
    }
    return result
  }, [])

  // ============================================================
  // Bước 8: Export performance metrics cho HUD
  // ============================================================
  const getPerfSnapshot = useCallback(() => ({ ...perfRef.current }), [])

  return {
    modelState,
    modelProgress,
    detections,
    lastResult,
    getTracksSnapshot,
    getPerfSnapshot,
  }
}
