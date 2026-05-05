# ============================================================
# aligner.py — FaceAligner (cv2 alignment)
# ============================================================
# Căn chỉnh khuôn mặt dựa trên vị trí 2 mắt từ YOLO keypoints.
# Xoay ảnh cho mặt thẳng trước khi đưa vào ArcFace.
#
# Chi phí: ~2-5ms/mặt
# ============================================================

import cv2
import numpy as np
import math

# Ngưỡng góc tối thiểu để kích hoạt xoay (độ)
MIN_ANGLE_TO_ALIGN = 1.0

# Tỷ lệ padding quanh mặt khi xoay
PADDING_RATIO = 0.5  # Thêm 50% mỗi chiều


class FaceAligner:
    """Căn chỉnh khuôn mặt dựa trên vị trí 2 mắt.

    Sử dụng:
        aligner = FaceAligner()
        aligned = aligner.align(frame, keypoints, box)
    """

    def __init__(self, min_angle=MIN_ANGLE_TO_ALIGN, padding_ratio=PADDING_RATIO):
        self._min_angle = min_angle
        self._padding_ratio = padding_ratio

    def align(self, frame, keypoints, box):
        """Xoay thẳng khuôn mặt dựa trên vị trí 2 mắt.

        Args:
            frame:     Ảnh gốc toàn khung hình (numpy array BGR).
            keypoints: Mảng 5 điểm mốc [(x,y), ...] hoặc None.
            box:       Bounding box (x1, y1, x2, y2).

        Returns:
            numpy array: Ảnh khuôn mặt đã xoay thẳng (BGR).
        """
        return align_face(frame, keypoints, box, self._min_angle, self._padding_ratio)


def align_face(frame, keypoints, box, min_angle=MIN_ANGLE_TO_ALIGN, padding_ratio=PADDING_RATIO):
    """Hàm standalone căn chỉnh khuôn mặt — có thể gọi trực tiếp không cần class.

    Args:
        frame:     Ảnh gốc toàn khung hình (numpy array BGR).
        keypoints: Mảng 5 điểm mốc từ YOLO [(x,y), ...].
                   Có thể là None nếu YOLO không trả về keypoints.
        box:       Bounding box (x1, y1, x2, y2) từ YOLO boxes.
        min_angle: Ngưỡng góc tối thiểu để xoay.
        padding_ratio: Tỷ lệ padding quanh mặt.

    Returns:
        numpy array: Ảnh khuôn mặt đã xoay thẳng (BGR).
                     Nếu không xoay được → trả về crop thô như cũ.
    """
    x1, y1, x2, y2 = box
    h_frame, w_frame = frame.shape[:2]

    # Đảm bảo box hợp lệ
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    # FALLBACK: Nếu không có keypoints → crop thô
    default_crop = frame[y1:y2, x1:x2]

    if keypoints is None or len(keypoints) < 2:
        return default_crop

    left_eye = keypoints[0]   # Mắt trái: (x, y)
    right_eye = keypoints[1]  # Mắt phải: (x, y)

    # Kiểm tra keypoints hợp lệ
    if (left_eye[0] == 0 and left_eye[1] == 0) or \
       (right_eye[0] == 0 and right_eye[1] == 0):
        return default_crop

    # Bước 1: Tính góc nghiêng giữa 2 mắt
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))

    # Nếu góc quá nhỏ → không cần xoay
    if abs(angle) < min_angle:
        return default_crop

    # Bước 2: Cắt vùng quanh mặt + padding
    face_w = x2 - x1
    face_h = y2 - y1
    pad = int(max(face_w, face_h) * padding_ratio)

    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w_frame, x2 + pad)
    py2 = min(h_frame, y2 + pad)

    padded_region = frame[py1:py2, px1:px2]

    # Bước 3: Xoay vùng đã cắt
    local_center = (
        (left_eye[0] + right_eye[0]) / 2.0 - px1,
        (left_eye[1] + right_eye[1]) / 2.0 - py1
    )

    M = cv2.getRotationMatrix2D(local_center, angle, 1.0)

    ph, pw = padded_region.shape[:2]
    rotated = cv2.warpAffine(
        padded_region, M, (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Bước 4: Tính lại tọa độ box sau khi xoay
    corners = np.array([
        [x1 - px1, y1 - py1],
        [x2 - px1, y1 - py1],
        [x2 - px1, y2 - py1],
        [x1 - px1, y2 - py1]
    ], dtype=np.float64)

    ones = np.ones((4, 1))
    corners_h = np.hstack([corners, ones])
    new_corners = (M @ corners_h.T).T

    nx1 = max(0, int(new_corners[:, 0].min()))
    ny1 = max(0, int(new_corners[:, 1].min()))
    nx2 = min(pw, int(new_corners[:, 0].max()))
    ny2 = min(ph, int(new_corners[:, 1].max()))

    # Bước 5: Crop mặt đã xoay thẳng
    aligned_face = rotated[ny1:ny2, nx1:nx2]

    if aligned_face.size == 0:
        return default_crop

    return aligned_face
