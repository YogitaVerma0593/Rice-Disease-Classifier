"""
Rice Leaf Disease Classifier - temporary local Streamlit app.

Run from your project directory with:
    streamlit run app.py

Uses the already-trained EfficientNet-B0 checkpoint. No training happens here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoints" / "efficientnet_b0_4class_best.pth"
TEST_CSV_PATH = BASE_DIR / "outputs" / "splits_4class" / "test.csv"

DEFAULT_CLASS_TO_IDX = {"Blast": 0, "Brown_Spot": 1, "Tungro": 2, "Healthy": 3}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

DISCLAIMER = (
    "**Disclaimer:** Predictions are experimental and may not generalize to "
    "real-world field conditions (different lighting, backgrounds, camera "
    "quality, growth stages, or diseases outside the trained classes). "
    "Do not use this tool as the sole basis for agricultural decisions."
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _norm(name):
    """Normalise a class name for tolerant matching."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _extract_class_to_idx(ckpt):
    """Return class_to_idx from the checkpoint if present, else the default."""
    if isinstance(ckpt, dict):
        mapping = None
        if isinstance(ckpt.get("class_to_idx"), dict):
            mapping = {str(k): int(v) for k, v in ckpt["class_to_idx"].items()}
        elif isinstance(ckpt.get("idx_to_class"), dict):
            mapping = {str(v): int(k) for k, v in ckpt["idx_to_class"].items()}
        else:
            for key in ("classes", "class_names"):
                if isinstance(ckpt.get(key), (list, tuple)):
                    mapping = {str(c): i for i, c in enumerate(ckpt[key])}
                    break
        if mapping and sorted(mapping.values()) == list(range(len(mapping))):
            return mapping, "checkpoint"
    return dict(DEFAULT_CLASS_TO_IDX), "default"


def _extract_state_dict(ckpt):
    """Find the model weights inside whatever the checkpoint contains."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                return value.state_dict()
            if isinstance(value, dict):
                return value
        # Plain state_dict (all values are tensors)
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not find model weights in the checkpoint file.")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Build EfficientNet-B0 (N classes), load the checkpoint, set eval mode."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(str(CHECKPOINT_PATH))

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # Checkpoint contains non-tensor objects; it's your own local file.
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    class_to_idx, mapping_source = _extract_class_to_idx(ckpt)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # weights=None: the checkpoint already holds all trained weights.
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_to_idx))

    state_dict = _extract_state_dict(ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    return model, idx_to_class, mapping_source


def get_model_or_stop():
    """Load the model; show a friendly message and stop the page on failure."""
    try:
        return load_model()
    except FileNotFoundError:
        st.error(
            f"Checkpoint not found:\n\n`{CHECKPOINT_PATH}`\n\n"
            "Make sure `app.py` is in your project root and the file exists."
        )
    except Exception as e:
        st.error(
            "The checkpoint was found but could not be loaded into "
            f"EfficientNet-B0 (4 classes).\n\n`{type(e).__name__}: {e}`"
        )
    st.stop()


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, images):
    """Return an (N, num_classes) array of softmax probabilities."""
    batch = torch.stack([TRANSFORM(img.convert("RGB")) for img in images]).to(DEVICE)
    logits = model(batch)
    return torch.softmax(logits, dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# Page: single-image prediction
# --------------------------------------------------------------------------
def page_predict():
    st.title("🌾 Rice Leaf Disease Classifier")
    st.caption("EfficientNet-B0 · Blast / Brown Spot / Tungro / Healthy")
    st.warning(DISCLAIMER)

    model, idx_to_class, _ = get_model_or_stop()

    # Changing the key of the uploader is the standard way to reset it.
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded = st.file_uploader(
        "Upload a rice leaf image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Please try another one.")
        return

    if st.button("🗑️ Clear image and prediction"):
        st.session_state.uploader_key += 1
        st.rerun()

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    probs = predict_probs(model, [image])[0]
    pred_idx = int(np.argmax(probs))

    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", idx_to_class[pred_idx].replace("_", " "))
        st.metric("Confidence", f"{probs[pred_idx] * 100:.2f}%")

        st.markdown("**Class probabilities**")
        for i in sorted(idx_to_class):
            st.write(f"{idx_to_class[i].replace('_', ' ')}: {probs[i] * 100:.2f}%")
            st.progress(float(probs[i]))


# --------------------------------------------------------------------------
# Page: evaluation on the existing test split
# --------------------------------------------------------------------------
def _guess_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return columns[0]


def _resolve_image_path(raw):
    """Try a few sensible locations for a path stored in test.csv."""
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p
    for root in (BASE_DIR, TEST_CSV_PATH.parent, BASE_DIR / "data"):
        candidate = root / p
        if candidate.exists():
            return candidate
    return None


def page_evaluate():
    st.title("📊 Test Set Evaluation")
    st.caption("Evaluates the saved model on `outputs/splits_4class/test.csv`. No retraining.")
    st.warning(DISCLAIMER)

    model, idx_to_class, _ = get_model_or_stop()
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    norm_to_idx = {_norm(k): v for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    if not TEST_CSV_PATH.exists():
        st.error(f"Test split not found:\n\n`{TEST_CSV_PATH}`")
        return

    try:
        df = pd.read_csv(TEST_CSV_PATH)
    except Exception as e:
        st.error(f"Could not read test.csv: {e}")
        return
    if df.empty:
        st.error("test.csv is empty.")
        return

    st.write(f"Found **{len(df)}** test images.")
    with st.expander("Preview test.csv"):
        st.dataframe(df.head(20), use_container_width=True)

    cols = list(df.columns)
    path_default = _guess_column(
        cols, ["filepath", "file_path", "path", "image_path", "image", "filename", "file"]
    )
    label_default = _guess_column(
        cols, ["label", "class", "class_name", "target", "category", "y"]
    )
    c1, c2 = st.columns(2)
    path_col = c1.selectbox("Image path column", cols, index=cols.index(path_default))
    label_col = c2.selectbox("Label column", cols, index=cols.index(label_default))

    use_all = st.checkbox("Evaluate all test images", value=True)
    if use_all:
        subset = df
    else:
        choices = df.index.tolist()
        selected = st.multiselect(
            "Select test images",
            choices,
            format_func=lambda i: f"{i}: {Path(str(df.loc[i, path_col])).name}",
        )
        subset = df.loc[selected]

    if not st.button("▶️ Run evaluation", type="primary"):
        return
    if subset.empty:
        st.warning("No images selected.")
        return

    # Collect valid (path, true label) pairs
    paths, y_true, skipped = [], [], 0
    for _, row in subset.iterrows():
        img_path = _resolve_image_path(row[path_col])
        label = row[label_col]
        if _norm(label) in norm_to_idx:
            true_idx = norm_to_idx[_norm(label)]
        else:
            try:
                true_idx = int(label)
            except (ValueError, TypeError):
                true_idx = None
        if img_path is None or true_idx not in idx_to_class:
            skipped += 1
            continue
        paths.append(img_path)
        y_true.append(true_idx)

    if skipped:
        st.warning(f"Skipped {skipped} row(s): image file missing or label not recognised.")
    if not paths:
        st.error("No valid images to evaluate. Check the column selections above.")
        return

    # Batched inference
    y_pred = []
    progress = st.progress(0.0, text="Evaluating...")
    batch_size = 32
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        probs = predict_probs(model, images)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        progress.progress(min((start + batch_size) / len(paths), 1.0))
    progress.empty()

    # Metrics
    labels = sorted(idx_to_class)
    acc = accuracy_score(y_true, y_pred)
    st.metric("Test accuracy", f"{acc * 100:.2f}%", help=f"{len(paths)} images evaluated")

    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Rice Leaf Disease Classifier", page_icon="🌾")

    page = st.sidebar.radio("Page", ["Predict", "Evaluate on test set (optional)"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: `{DEVICE}`")

    if CHECKPOINT_PATH.exists():
        try:
            _, idx_to_class, source = load_model()
            st.sidebar.caption(f"Class mapping from **{source}**:")
            st.sidebar.json({v: k for k, v in idx_to_class.items()})
        except Exception:
            pass  # the page itself will show a clear error

    if page == "Predict":
        page_predict()
    else:
        page_evaluate()


if __name__ == "__main__":
    main()