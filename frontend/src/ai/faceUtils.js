// ============================================================
// ai/faceUtils.js — Tiện ích xử lý khuôn mặt
// ============================================================
// NMS, IoU, crop face — tách ra để tái sử dụng.
// ============================================================

import * as ort from 'onnxruntime-web'

/**
 * Tính IoU (Intersection over Union) giữa 2 box.
 */
export function computeIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (intersection === 0) return 0

  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)

  return intersection / (areaA + areaB - intersection)
}

/**
 * Lọc danh sách detections bằng thuật toán NMS Greedy.
 */
export function nms(detections, iouThreshold) {
  if (detections.length === 0) return []

  detections.sort((a, b) => b.score - a.score)

  const kept = []
  const suppressed = new Set()

  for (let i = 0; i < detections.length; i++) {
    if (suppressed.has(i)) continue
    kept.push(detections[i])

    for (let j = i + 1; j < detections.length; j++) {
      if (suppressed.has(j)) continue
      if (computeIoU(detections[i].box, detections[j].box) > iouThreshold) {
        suppressed.add(j)
      }
    }
  }

  return kept
}

/**
 * Cắt 1 khuôn mặt từ video với padding ~40%, trả về Blob JPEG
 * và mảng keypoints đã re-map về tọa độ local trong crop.
 */
export async function cropFace(videoEl, detection, paddingRatio = 0.4) {
  const vw = videoEl.videoWidth
  const vh = videoEl.videoHeight
  const { box, keypoints } = detection

  const bw = box.x2 - box.x1
  const bh = box.y2 - box.y1
  const padW = bw * paddingRatio
  const padH = bh * paddingRatio

  const cropX1 = Math.max(0, Math.round(box.x1 - padW))
  const cropY1 = Math.max(0, Math.round(box.y1 - padH))
  const cropX2 = Math.min(vw, Math.round(box.x2 + padW))
  const cropY2 = Math.min(vh, Math.round(box.y2 + padH))
  const cropW = cropX2 - cropX1
  const cropH = cropY2 - cropY1

  if (cropW < 20 || cropH < 20) return null

  const MAX_DIM = 256
  const scale = Math.min(1, MAX_DIM / Math.max(cropW, cropH))
  const outW = Math.round(cropW * scale)
  const outH = Math.round(cropH * scale)

  const canvas = document.createElement('canvas')
  canvas.width = outW
  canvas.height = outH
  const ctx = canvas.getContext('2d')
  // Mirror theo chiều ngang (camera bị lật)
  ctx.translate(outW, 0)
  ctx.scale(-1, 1)
  ctx.drawImage(videoEl, cropX1, cropY1, cropW, cropH, 0, 0, outW, outH)

  // Re-map keypoints về tọa độ local (trong crop đã scale + mirror)
  const localKeypoints = []
  if (keypoints) {
    for (const [kx, ky] of keypoints) {
      const localX = Math.round((kx - cropX1) * scale)
      localKeypoints.push(
        outW - localX,            // mirror x
        Math.round((ky - cropY1) * scale),
      )
    }
  }

  // Chuyển canvas → JPEG blob (synchronous)
  const dataUrl = canvas.toDataURL('image/jpeg', 0.80)
  const byteString = atob(dataUrl.split(',')[1])
  const ab = new ArrayBuffer(byteString.length)
  const ia = new Uint8Array(ab)
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i)
  }
  const blob = new Blob([ab], { type: 'image/jpeg' })

  return { blob, localKeypoints, canvas }
}

/**
 * Lấy 1 frame từ <video> và chuyển sang tensor NCHW [1, 3, size, size].
 * Letterbox resize (giữ tỷ lệ, đệm pixel xám 114/255).
 */
export function preprocessFrame(videoEl, workCanvas, inputSize = 640) {
  const vw = videoEl.videoWidth
  const vh = videoEl.videoHeight

  workCanvas.width = inputSize
  workCanvas.height = inputSize
  const ctx = workCanvas.getContext('2d', { willReadFrequently: true })

  const scale = Math.min(inputSize / vw, inputSize / vh)
  const newW = Math.round(vw * scale)
  const newH = Math.round(vh * scale)
  const padX = (inputSize - newW) / 2
  const padY = (inputSize - newH) / 2

  ctx.fillStyle = 'rgb(114, 114, 114)'
  ctx.fillRect(0, 0, inputSize, inputSize)
  ctx.drawImage(videoEl, padX, padY, newW, newH)

  const imageData = ctx.getImageData(0, 0, inputSize, inputSize)
  const pixels = imageData.data

  const totalPixels = inputSize * inputSize
  const float32 = new Float32Array(3 * totalPixels)

  for (let i = 0; i < totalPixels; i++) {
    const base = i * 4
    float32[i]                  = pixels[base]     / 255.0
    float32[totalPixels + i]     = pixels[base + 1] / 255.0
    float32[2 * totalPixels + i] = pixels[base + 2] / 255.0
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, inputSize, inputSize])

  return { tensor, scale, padX, padY }
}
