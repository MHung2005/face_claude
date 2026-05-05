// ============================================================
// ai/yoloDetector.js — YOLOv12-face ONNX Detector
// ============================================================
// Load model ONNX, chạy inference trên browser (WebAssembly).
// ============================================================

import * as ort from 'onnxruntime-web'
import { nms, preprocessFrame } from './faceUtils'

// ── Cấu hình ──────────────────────────────────────────────────
const MODEL_URL = '/models/yolov12n-face.onnx'
const INPUT_SIZE = 640
const CONF_THRESHOLD = 0.45
const IOU_THRESHOLD = 0.5
const NUM_KEYPOINTS = 5
const DEBUG_YOLO = false

// ── Singleton session ─────────────────────────────────────────
let _session = null
let _loading = null

/**
 * Load model ONNX vào WASM session. Chỉ load 1 lần duy nhất.
 *
 * @param {function} onProgress - Callback (percent) báo tiến trình tải
 */
export async function loadModel(onProgress) {
  if (_session) return _session
  if (_loading) return _loading

  _loading = (async () => {
    try {
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
      ort.env.wasm.simd = true

      if (onProgress) onProgress(10)

      const response = await fetch(MODEL_URL)
      if (!response.ok) throw new Error(`Cannot fetch model: ${response.status}`)

      const contentLength = response.headers.get('Content-Length')
      const total = contentLength ? parseInt(contentLength) : 0
      const reader = response.body.getReader()

      const chunks = []
      let received = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        received += value.length
        if (onProgress && total > 0) {
          onProgress(10 + Math.round((received / total) * 70))
        }
      }

      const modelBuffer = new Uint8Array(received)
      let offset = 0
      for (const chunk of chunks) {
        modelBuffer.set(chunk, offset)
        offset += chunk.length
      }

      if (onProgress) onProgress(85)

      _session = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      })

      if (onProgress) onProgress(100)

      console.log('[YOLO-ONNX] Model loaded. Inputs:', _session.inputNames, 'Outputs:', _session.outputNames)
      return _session
    } catch (err) {
      _loading = null
      throw err
    }
  })()

  return _loading
}

/**
 * Kiểm tra model đã load chưa.
 */
export function isModelLoaded() {
  return _session !== null
}

/**
 * Chạy inference và trả về danh sách khuôn mặt.
 *
 * @param {HTMLVideoElement} videoEl
 * @param {HTMLCanvasElement} workCanvas
 * @returns {Promise<{detections: Array, timing: object}>}
 */
export async function detectFaces(videoEl, workCanvas) {
  if (!_session) return { detections: [], timing: null }

  const t0 = performance.now()
  const { tensor, scale, padX, padY } = preprocessFrame(videoEl, workCanvas, INPUT_SIZE)
  const t1 = performance.now()

  const inputName = _session.inputNames[0]
  const results = await _session.run({ [inputName]: tensor })
  const t2 = performance.now()

  const outputName = _session.outputNames[0]
  const output = results[outputName]
  const data = output.data
  const shape = output.dims

  if (DEBUG_YOLO) {
    console.info(`[YOLO model] output shape=${JSON.stringify(shape)}`)
  }

  const numDetections = shape[1]
  const numFeatures = shape[2]

  const candidates = []

  for (let d = 0; d < numDetections; d++) {
    const base = d * numFeatures

    const score = data[base + 4]
    if (score < CONF_THRESHOLD) continue

    const x1 = data[base + 0]
    const y1 = data[base + 1]
    const x2 = data[base + 2]
    const y2 = data[base + 3]

    const bx1 = (x1 - padX) / scale
    const by1 = (y1 - padY) / scale
    const bx2 = (x2 - padX) / scale
    const by2 = (y2 - padY) / scale

    const keypoints = []
    const kptBase = base + 5
    for (let k = 0; k < NUM_KEYPOINTS; k++) {
      const kx = data[kptBase + k * 3]
      const ky = data[kptBase + k * 3 + 1]
      keypoints.push([
        (kx - padX) / scale,
        (ky - padY) / scale,
      ])
    }

    candidates.push({ box: { x1: bx1, y1: by1, x2: bx2, y2: by2 }, score, keypoints })
  }

  const detections = nms(candidates, IOU_THRESHOLD)
  const t3 = performance.now()

  if (DEBUG_YOLO && detections.length > 0) {
    console.info(`[YOLO] raw=${candidates.length} → nms=${detections.length} | first box: x1=${detections[0].box.x1.toFixed(0)} y1=${detections[0].box.y1.toFixed(0)} score=${detections[0].score.toFixed(3)}`)
  }

  const timing = {
    preprocess_ms: Math.round((t1 - t0) * 10) / 10,
    inference_ms: Math.round((t2 - t1) * 10) / 10,
    postprocess_ms: Math.round((t3 - t2) * 10) / 10,
    total_ms: Math.round((t3 - t0) * 10) / 10,
  }

  return { detections, timing }
}
