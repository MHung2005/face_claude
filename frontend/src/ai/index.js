// ============================================================
// ai/index.js — Frontend AI Package (Public API)
// ============================================================
// Bất kỳ component nào cần AI chỉ cần:
//
//   import { detectFaces, cropFace, loadModel, isModelLoaded } from '../ai'
//
// ============================================================

export { loadModel, isModelLoaded, detectFaces } from './yoloDetector'
export { cropFace, computeIoU, nms, preprocessFrame } from './faceUtils'
