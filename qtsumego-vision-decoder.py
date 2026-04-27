import cv2
import numpy as np
import base64
import json
from typing import Dict, List

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def decode_tsumego_image(image_base64: str) -> Dict:
    """
    Decodes a 9x9 Go board image into a JSON structure for the Quantum Solver.
    Returns: {"board_size": 9, "black": [[r,c]...], "white": [[r,c]...], "empty": [[r,c]...]}
    """
    # 1. Decode Base64 to OpenCV Image
    image_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Find Board Boundary (works for screenshots & photos)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No board boundary detected")
        
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Fallback to bounding rect if contour isn't quadrilateral
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=np.float32)
        
    corners = order_points(corners)
    
    # 3. Perspective Transform (straighten to perfect 9x9)
    output_size = 540  # 60px per cell for clean sampling
    dst_corners = np.array([
        [0, 0], [output_size-1, 0],
        [output_size-1, output_size-1], [0, output_size-1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(img, M, (output_size, output_size))
    
    # 4. Extract 9x9 Grid & Classify Stones
    step = output_size / 8  # 8 intervals between 9 lines
    board_state = {"board_size": 9, "black": [], "white": [], "empty": []}
    
    # Sample board background color for relative thresholding
    bg_samples = []
    for r in range(9):
        for c in range(9):
            cx, cy = int(c * step), int(r * step)
            roi = warped[cy-10:cy+10, cx-10:cx+10]
            if roi.size > 0:
                bg_samples.append(np.mean(roi))
    board_bg = np.median(bg_samples, axis=0) if bg_samples else np.array([200, 180, 140])
    
    for r in range(9):
        for c in range(9):
            cx, cy = int(c * step), int(r * step)
            
            # Sample center 30% of cell (avoids grid lines)
            roi_size = int(step * 0.3)
            x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
            x2, y2 = min(output_size, cx + roi_size), min(output_size, cy + roi_size)
            roi = warped[y1:y2, x1:x2]
            
            if roi.size == 0:
                board_state["empty"].append([r, c])
                continue
                
            avg_color = np.mean(roi, axis=(0, 1))
            dist_to_bg = np.linalg.norm(avg_color - board_bg)
            
            # Classification logic (tune thresholds if needed)
            if dist_to_bg > 80 and np.mean(avg_color) < 90:
                board_state["black"].append([r, c])
            elif dist_to_bg > 60 and np.mean(avg_color) > 200:
                board_state["white"].append([r, c])
            else:
                board_state["empty"].append([r, c])
                
    return board_state