import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import easyocr

# ========== 1. Tải mô hình ==========
model_plate = YOLO(r"C:\Users\tuanl\Downloads\bienso\coco128\best1.pt")
model_corner = YOLO(r"C:\Users\tuanl\Downloads\bienso\coco128\best2.pt")

# ========== 2. Đọc ảnh gốc ==========
image_path = r"C:\Users\tuanl\Downloads\bienso\ccc_renamed\image_161.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Không thể đọc ảnh.")
height, width = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ========== 3. Phát hiện biển số ==========
results_plate = model_plate(img)
boxes_plate = results_plate[0].boxes.xyxy.cpu().numpy()
if len(boxes_plate) == 0:
    raise ValueError("Không phát hiện được biển số.")
x1, y1, x2, y2 = map(int, boxes_plate[0])
plate_crop = img_rgb[y1:y2, x1:x2]

# ========== 4. Phát hiện góc biển ==========
results_corners = model_corner(plate_crop)
boxes_corners = results_corners[0].boxes.xyxy.cpu().numpy()
if len(boxes_corners) < 4:
    raise ValueError("Không đủ 4 góc để xác định biển số.")

centers = []
for box in boxes_corners:
    x1c, y1c, x2c, y2c = box
    cx = int((x1c + x2c) / 2)
    cy = int((y1c + y2c) / 2)
    centers.append((cx, cy))

def sort_corners(pts):
    pts = sorted(pts, key=lambda x: x[1])
    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0])
    return [top[0], top[1], bottom[1], bottom[0]]

ordered_pts = sort_corners(centers)
src_pts = np.array(ordered_pts, dtype=np.float32)
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
w = int(max(distance(ordered_pts[0], ordered_pts[1]), distance(ordered_pts[2], ordered_pts[3])))
h = int(max(distance(ordered_pts[0], ordered_pts[3]), distance(ordered_pts[1], ordered_pts[2])))
dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

# ========== 5. Biến đổi phối cảnh ==========
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(plate_crop, M, (w, h))

# ========== 6. Làm sạch ảnh ==========
gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

# Morphology để loại nhiễu nhỏ
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# ========== 7. OCR ==========
reader = easyocr.Reader(['en'])
results = reader.readtext(cleaned)

def fix_ocr_text(text):
    return text.replace('O', '0').replace('o', '0').replace('I', '1')

plate_text = ""
if results:
    plate_text = ''.join(fix_ocr_text(res[1]) for res in results)
    print(f'Detected plate: "{plate_text}"')
else:
    print("Không nhận diện được ký tự nào.")

# ========== 8. Hiển thị ==========
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Ảnh gốc"); axs[0, 0].axis('off')
axs[0, 1].imshow(plate_crop); axs[0, 1].set_title("Biển số cắt"); axs[0, 1].axis('off')
axs[0, 2].imshow(warped); axs[0, 2].set_title("Sau trải phẳng"); axs[0, 2].axis('off')
axs[1, 0].imshow(gray, cmap='gray'); axs[1, 0].set_title("Ảnh xám"); axs[1, 0].axis('off')
axs[1, 1].imshow(cleaned, cmap='gray'); axs[1, 1].set_title("Sau xử lý (morph)"); axs[1, 1].axis('off')
axs[1, 2].axis('off')
axs[1, 2].text(0.5, 0.5, f"Biển số: {plate_text}" if plate_text else "Không nhận diện được",
              fontsize=18, ha='center', va='center', color='blue' if plate_text else 'red')

plt.tight_layout()
plt.show()
