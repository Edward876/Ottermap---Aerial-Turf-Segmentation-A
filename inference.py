import os
import sys
import json
import argparse
import tempfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import rasterio
    import rasterio.transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from shapely.geometry import Polygon, mapping
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# -------------------------------------------------------
TILE_SIZE       = 512
STRIDE          = 256
THRESHOLD       = 0.50
MEAN            = (0.485, 0.456, 0.406)
STD             = (0.229, 0.224, 0.225)
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "best_model_segformer.pth")
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
# -------------------------------------------------------

transform = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print(f"ERROR: model weights not found at {weights_path}")
        sys.exit(1)
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels              = 1,
        ignore_mismatched_sizes = True,
    )
    m.load_state_dict(torch.load(weights_path, map_location=device))
    return m.to(device).eval()


def resolve_source(source):
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  downloading: {source}")
        ext = os.path.splitext(source.split("?")[0])[-1] or ".tif"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        urllib.request.urlretrieve(source, tmp.name)
        print(f"  saved to: {tmp.name}")
        return tmp.name
    return source


def read_image(path):
    geo_transform, crs = None, None
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                img = src.read([1,2,3]).transpose(1,2,0)
                if src.crs:
                    geo_transform = src.transform
                    crs           = src.crs.to_epsg()
                return img.astype(np.uint8), geo_transform, crs
        except Exception:
            pass
    img = np.array(Image.open(path).convert("RGB"))
    return img, geo_transform, crs


def predict(model, img, device, threshold=THRESHOLD):
    h, w   = img.shape[:2]
    oh, ow = h, w

    if h < TILE_SIZE or w < TILE_SIZE:
        scale  = TILE_SIZE / min(h, w)
        nh, nw = int(h*scale)+1, int(w*scale)+1
        img    = cv2.resize(img, (nw, nh))
        print(f"  resized {ow}x{oh} -> {nw}x{nh}")
        h, w   = img.shape[:2]

    pad_h = (TILE_SIZE - h % STRIDE) % STRIDE
    pad_w = (TILE_SIZE - w % STRIDE) % STRIDE
    if pad_h > 0 or pad_w > 0:
        img  = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
        h, w = img.shape[:2]

    full_mask = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    ys        = list(range(0, h - TILE_SIZE + 1, STRIDE))
    xs        = list(range(0, w - TILE_SIZE + 1, STRIDE))
    total, done = len(ys)*len(xs), 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                    tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))

                variants = [
                    tile,
                    tile[:,::-1].copy(),
                    tile[::-1,:].copy(),
                    tile[::-1,::-1].copy(),
                ]
                preds = []
                for v in variants:
                    t      = transform(image=v)["image"].unsqueeze(0).to(device)
                    logits = model(pixel_values=t).logits
                    up     = F.interpolate(logits, size=(TILE_SIZE, TILE_SIZE),
                                           mode="bilinear", align_corners=False)
                    pred   = torch.sigmoid(up).squeeze().cpu().numpy()
                    preds.append(pred)

                preds[1] = preds[1][:,::-1]
                preds[2] = preds[2][::-1,:]
                preds[3] = preds[3][::-1,::-1]

                full_mask[y:y+TILE_SIZE, x:x+TILE_SIZE] += np.mean(preds, axis=0)
                count_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1
                done += 1
                print(f"  tiles: {done}/{total}", end="\r")
    print()

    count_map = np.maximum(count_map, 1)
    result    = ((full_mask / count_map) > threshold).astype(np.uint8) * 255
    result    = result[:min(oh, result.shape[0]), :min(ow, result.shape[1])]
    if result.shape[:2] != (oh, ow):
        result = cv2.resize(result, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return result


def clean_mask(mask, min_area=2000, closing_kernel=15):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def make_overlay(img, mask):
    out        = img.copy().astype(np.float32)
    alpha      = (mask > 0).astype(np.float32) * 0.5
    out[:,:,0] = np.clip(img[:,:,0] * (1-alpha),             0, 255)
    out[:,:,1] = np.clip(img[:,:,1] * (1-alpha) + 180*alpha, 0, 255)
    out[:,:,2] = np.clip(img[:,:,2] * (1-alpha),             0, 255)
    return out.astype(np.uint8)


def mask_to_geojson(mask, geo_transform=None, crs=None, img_w=None, img_h=None):
    if not HAS_SHAPELY:
        print("  warning: shapely not installed, skipping GeoJSON output")
        return {"type": "FeatureCollection", "features": []}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features    = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        pts = cnt.squeeze(1)
        if len(pts) < 3:
            continue
        if geo_transform is not None and HAS_RASTERIO:
            coords = [rasterio.transform.xy(geo_transform, p[1], p[0]) for p in pts]
        else:
            coords = [(float(p[0])/img_w, float(p[1])/img_h) for p in pts]
        try:
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                features.append({
                    "type"      : "Feature",
                    "properties": {"id": len(features), "area": round(poly.area, 6)},
                    "geometry"  : mapping(poly)
                })
        except Exception:
            continue

    return {
        "type"    : "FeatureCollection",
        "crs"     : {"type": "name", "properties": {
            "name": f"EPSG:{crs}" if crs else "urn:ogc:def:crs:OGC:1.3:CRS84"
        }},
        "features": features
    }


def run(image_path, output_dir, threshold, model_path):
    device = get_device()
    print(f"\n  device      : {device}")

    image_path = resolve_source(image_path)
    base       = os.path.splitext(os.path.basename(image_path.split("?")[0]))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"  loading     : {image_path}")
    img, geo_transform, crs = read_image(image_path)
    h, w = img.shape[:2]
    print(f"  size        : {w}x{h}")
    print(f"  georef      : {geo_transform is not None}")

    print(f"  loading model from {model_path}...")
    model = load_model(model_path, device)

    print(f"  predicting (threshold={threshold})...")
    raw_mask = predict(model, img, device, threshold=threshold)

    print(f"  cleaning mask...")
    mask    = clean_mask(raw_mask)
    overlay = make_overlay(img, mask)
    cov     = 100*(mask>0).sum()/(h*w)

    out_overlay = os.path.join(output_dir, f"{base}_overlay.png")
    out_mask    = os.path.join(output_dir, f"{base}_mask.png")
    out_geojson = os.path.join(output_dir, f"{base}_predictions.geojson")

    Image.fromarray(overlay).save(out_overlay)
    Image.fromarray(mask).save(out_mask)

    gj = mask_to_geojson(mask, geo_transform, crs, w, h)
    with open(out_geojson, "w") as f:
        json.dump(gj, f, indent=2)

    n_poly = len(gj["features"])
    print(f"\n  coverage    : {cov:.1f}%")
    print(f"  polygons    : {n_poly}")
    print(f"\n  outputs:")
    print(f"    {out_overlay}")
    print(f"    {out_mask}")
    print(f"    {out_geojson}")


def main():
    parser = argparse.ArgumentParser(description="Turf segmentation - Ottermap challenge")
    parser.add_argument("--image",     required=True,      help="path or URL to input image (.tif/.jpg/.png)")
    parser.add_argument("--output",    default="results",  help="output directory (default: results)")
    parser.add_argument("--threshold", default=THRESHOLD,  type=float, help=f"prediction threshold (default: {THRESHOLD})")
    parser.add_argument("--model",     default=MODEL_PATH, help="path to model weights")
    args = parser.parse_args()

    print("=" * 50)
    print("  Turf Segmentation - Ottermap Challenge")
    print("  Model : SegFormer mit_b2")
    print("  IoU   : 0.5796 (true generalization, unseen parcel)")
    print("=" * 50)

    run(
        image_path = args.image,
        output_dir = args.output,
        threshold  = args.threshold,
        model_path = args.model,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()