# %% STEP 1: Import Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass, shift as nd_shift

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
SHOW_PLOTS = (__name__ == "__main__")


def show():
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:               # interactive cell environments
    BASE_DIR = os.getcwd()


# %% STEP 2: Set Random Seed

S = 1187
np.random.seed(S)

print("Libraries imported successfully!")
print("Random Seed:", S)


# %% STEP 3: Load Sklearn Digits Dataset

digits = load_digits()

X = digits.data
y = digits.target

print("\nDataset loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total images:", len(X))
print("Unique digits:", np.unique(y))


# %% STEP 4: Display First 10 Sklearn Images

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")

plt.tight_layout()
show()


# %% STEP 5: Split Sklearn Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=S,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# %% STEP 6: Load Handwriting Image

img = cv2.imread(os.path.join(BASE_DIR, "handwriting digit.jpeg"))

if img is None:
    raise FileNotFoundError(
        "Check handwriting digit.jpeg filename and folder!"
    )

print("\nHandwriting image loaded!")
print("Image dimensions:", img.shape)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 10))
plt.imshow(img_rgb)
plt.title("Original Handwriting")
plt.axis("off")
show()


# %% STEP 7: Convert Image to Grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 10))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
show()


# %% STEP 8: Apply Gaussian Blur and Thresholding

blur = cv2.GaussianBlur(gray, (5, 5), 0)

ink = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    15
)

plt.figure(figsize=(8, 10))
plt.imshow(ink, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
show()


# %% STEP 9: Merge Broken Handwriting Strokes

kernel = np.ones((5, 5), np.uint8)

merged = cv2.morphologyEx(
    ink,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

plt.figure(figsize=(8, 10))
plt.imshow(merged, cmap="gray")
plt.title("Merged Handwriting")
plt.axis("off")
show()

print("Image preprocessing completed!")


# %% STEP 10: Detect Handwritten Regions

H, W = gray.shape

contours, _ = cv2.findContours(
    merged,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

boxes = []

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    # Ignore very small regions
    if w < 8 or h < 15:
        continue

    # Ignore image border regions
    margin = 0.02

    if (
        x < margin * W
        or y < margin * H
        or x + w > (1 - margin) * W
        or y + h > (1 - margin) * H
    ):
        continue

    boxes.append((x, y, w, h))

print("Original detected contours:", len(boxes))

# Fifth zero of the first row touches the top border, so the border filter
# above rejects it. Its box is added manually.
missing_zero_box = (800, 0, 180, 120)


# %% STEP 11: Remove Roll Number, Add Missing Zero, Order Digits Row by Row

digit_boxes = [
    (x, y, w, h) for (x, y, w, h) in boxes
    if not (x > 1100 and y < 250)          # roll number area
]

print("Regions after removing roll number:", len(digit_boxes))

all_boxes = digit_boxes + [missing_zero_box]
print("Total regions including missing zero:", len(all_boxes))

assert len(all_boxes) == 50, (
    f"Expected 50 digit regions, got {len(all_boxes)}. "
    "Check thresholding / roll number removal."
)

# The sheet has 10 rows (digits 0..9) with 5 samples each.
# Group boxes into rows by their vertical centre (NOT by top edge),
# then sort each row left to right.
centers_y = np.array([[b[1] + b[3] / 2] for b in all_boxes])

km = KMeans(n_clusters=10, n_init=10, random_state=S).fit(centers_y)
row_order = np.argsort(km.cluster_centers_.ravel())

rows = []
for c in row_order:
    members = [all_boxes[i] for i in np.where(km.labels_ == c)[0]]
    rows.append(sorted(members, key=lambda b: b[0]))

if [len(r) for r in rows] != [5] * 10:
    print("WARNING: row clustering did not give 5 digits per row "
          f"({[len(r) for r in rows]}). Falling back to sorting by y and "
          "cutting into rows of 5. CHECK the preview in Step 12!")
    by_y = sorted(all_boxes, key=lambda b: b[1] + b[3] / 2)
    rows = [sorted(by_y[r * 5:(r + 1) * 5], key=lambda b: b[0])
            for r in range(10)]

ordered_boxes = [b for row in rows for b in row]

# Row-major order: five 0s, five 1s, ..., five 9s
handwritten_labels = np.repeat(np.arange(10), 5)

print("Digits per row:", [len(r) for r in rows])


# %% STEP 12: Display Detected Regions With Their Labels

preview = img_rgb.copy()

for i, (x, y, w, h) in enumerate(ordered_boxes):

    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.putText(
        preview,
        str(handwritten_labels[i]),
        (x, max(y - 8, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 160, 0),
        3
    )

plt.figure(figsize=(10, 13))
plt.imshow(preview)
plt.title("Detected Regions (green number = assigned label)")
plt.axis("off")
show()


# %% STEP 13: Display All 50 Crops With Their Labels

def get_mask_crop(box):
    """Binary ink crop (white ink, black background) for one box."""
    x, y, w, h = box
    crop = merged[y:y + h, x:x + w].copy()

    if box == missing_zero_box:
        # keep only the biggest ink blob (drops paper-edge noise)
        cnts, _ = cv2.findContours(
            crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) == 0:
            raise ValueError("No ink found in the missing zero box. "
                             "Adjust missing_zero_box.")
        cnt = max(cnts, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        crop = crop[cy:cy + ch, cx:cx + cw]

    return crop


plt.figure(figsize=(16, 9))

for i, box in enumerate(ordered_boxes):
    plt.subplot(5, 10, i + 1)
    plt.imshow(get_mask_crop(box), cmap="gray")
    plt.title(f"#{i+1} -> {handwritten_labels[i]}", fontsize=9)
    plt.axis("off")

plt.tight_layout()
show()


# %% STEP 14: Convert Each Digit Into a sklearn-style 8x8 Image
# sklearn digits: thick strokes, bright digit on dark background,
# aspect ratio preserved, about 1 pixel of margin, values 0-16.

TARGET_H = 7          # max digit height in pixels (sklearn digits use ~full height)
TARGET_W = 5          # max digit width in pixels (sklearn keeps 1px side margins)
STROKE_FRAC = 0.14    # extra stroke thickness as a fraction of digit size
                      # (increase if digits look too thin, decrease if blobs)
BLUR_SIGMA = 0.7      # softness; sklearn digits are blurry (0 = no blur)
CENTER_MASS = True    # shift each digit so its centre of mass is the image centre


def mask_to_8x8(mask):
    h, w = mask.shape
    size = max(h, w)

    # pad so dilation does not get clipped at the crop edges
    pad = size // 4 + 2
    padded = cv2.copyMakeBorder(
        mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )

    # thicken strokes to sklearn-like width
    k = max(3, int(STROKE_FRAC * size) | 1)
    padded = cv2.dilate(
        padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    # tight bounding box around the (thickened) digit
    ys, xs = np.nonzero(padded)
    padded = padded[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # resize keeping the aspect ratio (no stretching)
    h, w = padded.shape
    scale = min(TARGET_H / h, TARGET_W / w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    small = cv2.resize(
        padded, (new_w, new_h), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    # centre inside an 8x8 canvas
    canvas = np.zeros((8, 8), dtype=np.float32)
    y0 = (8 - new_h) // 2
    x0 = (8 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small

    # soften like sklearn's block-averaged digits
    if BLUR_SIGMA > 0:
        canvas = cv2.GaussianBlur(canvas, (3, 3), BLUR_SIGMA)

    # centre of mass -> middle of the 8x8 grid (integer shift, no extra blur)
    if CENTER_MASS:
        cy, cx = center_of_mass(canvas)
        canvas = nd_shift(
            canvas, (round(3.5 - cy), round(3.5 - cx)),
            order=0, mode="constant", cval=0
        )

    # scale to 0-16 like sklearn
    canvas = canvas / canvas.max() * 16
    return np.round(canvas).astype(np.uint8)


handwritten_images = np.array([
    mask_to_8x8(get_mask_crop(box)).flatten() for box in ordered_boxes
])

print("Final handwritten images:", len(handwritten_images))
print("Final dataset shape:", handwritten_images.shape)


# %% STEP 15: Display All 50 Handwritten 8x8 Images

plt.figure(figsize=(15, 8))

for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(
        handwritten_images[i].reshape(8, 8),
        cmap="gray",
        vmin=0,
        vmax=16
    )
    plt.title(f"#{i+1} -> {handwritten_labels[i]}", fontsize=9)
    plt.axis("off")

plt.tight_layout()
show()


# %% STEP 15b: My 50 Digits Next to 10 load_digits Digits
# Top row: one sklearn digit per class (0-9).
# Rows below: my 5 samples of each digit (column = digit).

first_idx = [int(np.where(digits.target == d)[0][0]) for d in range(10)]

fig, axes = plt.subplots(6, 10, figsize=(14, 9))

for d in range(10):
    axes[0, d].imshow(digits.images[first_idx[d]], cmap="gray", vmin=0, vmax=16)
    axes[0, d].set_title(f"{d}", fontsize=10)
    for s in range(5):
        axes[s + 1, d].imshow(
            handwritten_images[d * 5 + s].reshape(8, 8),
            cmap="gray", vmin=0, vmax=16
        )

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel("load_digits", fontsize=9)
for s in range(5):
    axes[s + 1, 0].set_ylabel(f"mine #{s+1}", fontsize=9)

plt.suptitle("load_digits (top row) vs my 50 handwritten digits")
plt.tight_layout()
show()


# %% STEP 16: Verify Final Dataset

print("Labels:", handwritten_labels)
print("Labels shape:", handwritten_labels.shape)

print("\nFinal Dataset Verification")
print("Images:", handwritten_images.shape)
print("Labels:", handwritten_labels.shape)
print("Minimum pixel value:", handwritten_images.min())
print("Maximum pixel value:", handwritten_images.max())
print("Mean pixel (sklearn train):", round(float(X_train.mean()), 2))
print("Mean pixel (handwritten):  ", round(float(handwritten_images.mean()), 2))

assert handwritten_images.shape == (50, 64), (
    "Expected 50 images. Check contour detection and zero crop."
)
assert handwritten_labels.shape == (50,)
assert len(handwritten_images) == len(handwritten_labels)

print("Dataset verification successful!")

# Visual check: tune TARGET_H / TARGET_W / STROKE_FRAC / BLUR_SIGMA so that
# row 2 looks like row 1 (compare with sklearn, NOT with test accuracy).
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for d in range(10):
    axes[0, d].imshow(X_train[y_train == d].mean(axis=0).reshape(8, 8),
                      cmap="gray", vmin=0, vmax=16)
    axes[1, d].imshow(handwritten_images[handwritten_labels == d].mean(axis=0).reshape(8, 8),
                      cmap="gray", vmin=0, vmax=16)
    axes[2, d].imshow(handwritten_images[d * 5].reshape(8, 8),
                      cmap="gray", vmin=0, vmax=16)
axes[0, 0].set_ylabel("sklearn avg")
axes[1, 0].set_ylabel("mine avg")
axes[2, 0].set_ylabel("mine #1")
plt.tight_layout()
show()


# %% STEP 17: Train and Evaluate ML Models (runs only when Part_A is run directly)

def run_ml_models():

    models = {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=S)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=S
        )
    }

    results = {}
    all_predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        sklearn_accuracy = accuracy_score(y_test, model.predict(X_test))

        hw_pred = model.predict(handwritten_images)
        hw_accuracy = accuracy_score(handwritten_labels, hw_pred)

        results[name] = {
            "Sklearn Accuracy": sklearn_accuracy,
            "Handwritten Accuracy": hw_accuracy
        }
        all_predictions[name] = hw_pred

        print("\n================================")
        print("MODEL:", name)
        print("================================")
        print("Sklearn Test Accuracy:", round(sklearn_accuracy * 100, 2), "%")
        print("Handwritten Accuracy:", round(hw_accuracy * 100, 2), "%")

        print("\nClassification Report:")
        print(classification_report(
            handwritten_labels, hw_pred, zero_division=0
        ))

        cm = confusion_matrix(
            handwritten_labels, hw_pred, labels=np.arange(10)
        )
        print("\nHandwritten Confusion Matrix:")
        print(cm)

        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=np.arange(10)
        ).plot(cmap="Blues")
        plt.title(f"{name} - Handwritten Confusion Matrix")
        plt.show()

    # Final comparison (printed once)
    print("\n========== FINAL MODEL COMPARISON ==========")
    for name, scores in results.items():
        print("\nModel:", name)
        print("Sklearn Accuracy:",
              round(scores["Sklearn Accuracy"] * 100, 2), "%")
        print("Handwritten Accuracy:",
              round(scores["Handwritten Accuracy"] * 100, 2), "%")

    # Actual vs predicted for the best model on the handwritten digits
    best = max(results, key=lambda n: results[n]["Handwritten Accuracy"])
    print(f"\nActual vs Predicted Digits ({best})")
    for i in range(50):
        print(f"Image {i+1}: Actual = {handwritten_labels[i]}, "
              f"Predicted = {all_predictions[best][i]}")


if __name__ == "__main__":
    run_ml_models()