"""
Camouflaged Object Detection — Gradio Space
CODNet: convnext_small + CBAM + FAM + SAM
HF username: penguin18
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from model import CODNet

# ─────────────────────────────────────────────────────────────
# 1.  LOAD MODEL
# ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = hf_hub_download(
    repo_id="penguin18/codnet-weights",
    filename="checkpoint.pth",
    repo_type="model",
)

model = CODNet().to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state)
model.eval()
print(f"Model loaded on {DEVICE}")

# ─────────────────────────────────────────────────────────────
# 2.  PRE/POST PROCESSING
# ─────────────────────────────────────────────────────────────

IMG_SIZE = 448

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict_tta(mdl, img_tensor):
    with torch.no_grad():
        p0, *_ = mdl(img_tensor)
        p1, *_ = mdl(torch.flip(img_tensor, dims=[3]))
        p1     = torch.flip(p1, dims=[3])
        p2, *_ = mdl(torch.flip(img_tensor, dims=[2]))
        p2     = torch.flip(p2, dims=[2])
        return torch.sigmoid((p0 + p1 + p2) / 3.0)


def adaptive_threshold(p_np):
    p_u8 = (p_np * 255).astype(np.uint8)
    _, binary = cv2.threshold(p_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ratio = binary.mean() / 255.0
    if ratio < 0.005 or ratio > 0.98:
        binary = (p_u8 > int(0.35 * 255)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


# ─────────────────────────────────────────────────────────────
# 3.  INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────

def detect(pil_image, overlay_alpha, overlay_color):
    if pil_image is None:
        return None, None, None, "Please upload an image."

    pil_image = pil_image.convert("RGB")
    orig_w, orig_h = pil_image.size

    img_tensor = img_transform(pil_image).unsqueeze(0).to(DEVICE)
    pred       = predict_tta(model, img_tensor)
    pred       = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    p_np = pred[0, 0].cpu().numpy()
    p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8)

    binary    = adaptive_threshold(p_np)
    pred_mask = (binary > 127).astype(np.float32)

    # ── Heatmap ──────────────────────────────────────────────
    heatmap_u8     = (p_np * 255).astype(np.uint8)
    heatmap_colour = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_INFERNO)
    heatmap_pil    = Image.fromarray(cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB))

    # ── Binary mask ──────────────────────────────────────────
    mask_pil = Image.fromarray(binary)

    # ── Coloured overlay ─────────────────────────────────────
    color_map = {
        "Green":  (0, 1, 0),
        "Red":    (1, 0, 0),
        "Blue":   (0, 0, 1),
        "Yellow": (1, 1, 0),
    }
    r, g, b   = color_map.get(overlay_color, (0, 1, 0))
    orig_np   = np.array(pil_image).astype(np.float32)
    overlay   = orig_np.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + r * overlay_alpha * 255 * pred_mask, 0, 255)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + g * overlay_alpha * 255 * pred_mask, 0, 255)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + b * overlay_alpha * 255 * pred_mask, 0, 255)
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))

    # ── Stats ────────────────────────────────────────────────
    coverage   = pred_mask.mean() * 100
    num_pixels = int(pred_mask.sum())

    # Bounding box of detected region
    ys, xs = np.where(pred_mask > 0)
    if len(xs) > 0:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        box_w  = x2 - x1
        box_h  = y2 - y1
        region = f"{box_w}×{box_h} px  (at x={x1}, y={y1})"
    else:
        region = "none"

    info = (
        f"Camouflaged area  :  {coverage:.1f}% of image\n"
        f"Detected pixels   :  {num_pixels:,}\n"
        f"Bounding region   :  {region}\n"
        f"Image size        :  {orig_w}×{orig_h} px"
    )
    return heatmap_pil, mask_pil, overlay_pil, info


# ─────────────────────────────────────────────────────────────
# 4.  GRADIO UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Camouflaged Object Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🦎 Camouflaged Object Detection
        ### CODNet — ConvNeXt-Small · CBAM · FAM · SAM · 3-way TTA · 50 epochs
        Upload any natural scene image. The model will locate hidden camouflaged objects
        and show you a **heatmap**, **binary mask**, and a **highlighted overlay**.
        > Trained on COD10K · CAMO · CHAMELEON datasets
        """
    )

    with gr.Row():

        # ── LEFT: inputs ─────────────────────────────────────
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="📷 Input image")

            with gr.Group():
                gr.Markdown("**Overlay settings**")
                overlay_color = gr.Radio(
                    choices=["Green", "Red", "Blue", "Yellow"],
                    value="Green",
                    label="Highlight colour"
                )
                overlay_alpha = gr.Slider(
                    minimum=0.1, maximum=0.8, value=0.35, step=0.05,
                    label="Highlight strength"
                )

            run_btn = gr.Button("🔍  Detect", variant="primary", size="lg")

        # ── RIGHT: outputs ───────────────────────────────────
        with gr.Column(scale=2):
            with gr.Row():
                out_heat    = gr.Image(type="pil", label="🌡️ Probability heatmap")
                out_mask    = gr.Image(type="pil", label="⬜ Binary mask")
                out_overlay = gr.Image(type="pil", label="🎨 Overlay")

            out_info = gr.Textbox(label="📊 Detection stats", lines=4)

    run_btn.click(
        fn=detect,
        inputs=[inp_image, overlay_alpha, overlay_color],
        outputs=[out_heat, out_mask, out_overlay, out_info],
    )

    gr.Markdown(
        """
        ---
        **How to read the outputs:**
        - 🌡️ **Heatmap** — brighter = model is more confident something is hidden there
        - ⬜ **Binary mask** — white pixels = detected camouflaged object
        - 🎨 **Overlay** — your chosen highlight colour drawn over the detected region
        **Architecture:** ConvNeXt-Small backbone · CBAM channel+spatial attention ·
        FAM hierarchical feature fusion · SAM multi-scale dilated convolutions ·
        Lovász + wBCE + Dice + IoU losses · deep supervision
        """
    )

demo.launch()
