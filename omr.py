import cv2
import numpy as np


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Locates the largest quadrilateral contour (e.g., a phone-scanned sheet)
    and warps the perspective to align and flatten it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection to locate document borders
    edged = cv2.Canny(blurred, 50, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If the contour has 4 points, we found the page border
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            
            # Sort the points: [top-left, top-right, bottom-right, bottom-left]
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            # Check for skew: calculate horizontal and vertical deviation of sides
            dev_left_x = abs(rect[0][0] - rect[3][0])
            dev_right_x = abs(rect[1][0] - rect[2][0])
            dev_top_y = abs(rect[0][1] - rect[1][1])
            dev_bottom_y = abs(rect[3][1] - rect[2][1])
            
            h, w = image.shape[:2]
            threshold_w = w * 0.015
            threshold_h = h * 0.015
            
            # Only warp if there is significant skew/tilt in horizontal or vertical lines
            if (max(dev_left_x, dev_right_x) > threshold_w) or (max(dev_top_y, dev_bottom_y) > threshold_h):
                return _warp_document(image, pts)
            
    return image


def _warp_document(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warps a 4-point contour to retrieve a straightened top-down perspective."""
    # Order points: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # Compute widths and heights of the warped image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_omr_sheet(image: np.ndarray) -> str:
    """
    Parses a warped bubble sheet image using adaptive thresholding
    and contour geometry filtering. Groups bubble outlines into rows
    and maps filled marks to answers.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to remove hand shadows and illumination gradients
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Locate all bubble/checkbox candidates
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Keep contours that are roughly square/circular and match expected sizing
        if 12 <= w <= 80 and 12 <= h <= 80 and 0.75 <= ar <= 1.35:
            # Solidity filter: checks density relative to bounding box
            area = cv2.contourArea(c)
            box_area = w * h
            solidity = area / float(box_area) if box_area > 0 else 0
            
            if solidity > 0.6:
                bubbles.append((x, y, w, h, c))

    if not bubbles:
        return "No OMR checkboxes or bubbles detected on the sheet."

    # Group detected bubbles into horizontal rows (questions)
    # Sort top-to-bottom first
    bubbles = sorted(bubbles, key=lambda b: b[1])
    
    rows = []
    current_row = []
    last_y = -1
    
    # Define a vertical clustering threshold based on average bubble height
    avg_h = sum(b[3] for b in bubbles) / len(bubbles)
    y_threshold = avg_h * 0.8
    
    for b in bubbles:
        y = b[1]
        if last_y == -1 or abs(y - last_y) <= y_threshold:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
        last_y = y
    if current_row:
        rows.append(current_row)

    # Filter out rows that do not have enough choices to represent a question (e.g. isolated noise)
    valid_rows = []
    for r in rows:
        if len(r) >= 2: # At least a True/False or A/B/C/D grid row
            # Sort left-to-right
            r_sorted = sorted(r, key=lambda b: b[0])
            valid_rows.append(r_sorted)
            
    # Sort valid rows top-to-bottom
    valid_rows = sorted(valid_rows, key=lambda r: r[0][1])

    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    results = []
    
    for i, row in enumerate(valid_rows, start=1):
        marked_choices = []
        for col_idx, (x, y, w, h, c) in enumerate(row):
            # Crop bubble ROI from thresholded image
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Extract white pixel count inside the contour mask
            roi = cv2.bitwise_and(thresh, thresh, mask=mask)
            total_pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(roi)
            
            ratio = filled_pixels / float(total_pixels) if total_pixels > 0 else 0
            
            # If bubble is more than 40% filled, classify as marked
            if ratio >= 0.40:
                opt_letter = options[col_idx] if col_idx < len(options) else f"Col{col_idx+1}"
                marked_choices.append(opt_letter)
        
        answer = ", ".join(marked_choices) if marked_choices else "Unmarked"
        results.append(f"Question {i:02d}: [{answer}]")

    return "\n".join(results)
