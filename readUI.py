# tft_screenshot_to_json.py
# Usage:
#   python tft_screenshot_to_json.py --image "vlcsnap.png" --out "tft.json" --debug_dir "debug"
#
# Requirements:
#   pip install opencv-python pillow pytesseract rapidfuzz
#   + installer Tesseract OCR:
#     - Windows: https://github.com/UB-Mannheim/tesseract/wiki
#     - Linux: sudo apt-get install tesseract-ocr
#     - macOS: brew install tesseract

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz


# -------------------------
# OCR helpers
# -------------------------

@dataclass
class OCRResult:
    text: str
    confidence: float
    variant: str


def _avg_conf_from_data(data: Dict[str, List[str]]) -> float:
    confs = []
    for c in data.get("conf", []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except Exception:
            pass
    if not confs:
        return -1.0
    return float(sum(confs) / len(confs))


def ocr_with_conf(img: np.ndarray, config: str) -> OCRResult:
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(img, config=config) or ""
    text = text.strip()
    conf = _avg_conf_from_data(data)
    return OCRResult(text=text, confidence=conf, variant=config)


def preprocess_variants(roi_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Return multiple preprocessed variants to improve OCR robustness."""
    variants = []

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    variants.append(("gray_up", up))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v2 = clahe.apply(up)
    variants.append(("clahe", v2))

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    v3 = cv2.filter2D(v2, -1, kernel)
    variants.append(("clahe_sharpen", v3))

    blur = cv2.GaussianBlur(v2, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", th))
    variants.append(("otsu_inv", 255 - th))

    ad = cv2.adaptiveThreshold(v2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    variants.append(("adaptive", ad))
    variants.append(("adaptive_inv", 255 - ad))

    return variants


def best_ocr(roi_bgr: np.ndarray, psm: int, whitelist: Optional[str] = None) -> OCRResult:
    """Try multiple preprocessing variants and keep the OCR with best confidence."""
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"

    best = OCRResult(text="", confidence=-999, variant="")

    for name, v in preprocess_variants(roi_bgr):
        res = ocr_with_conf(v, cfg)
        score = res.confidence
        if res.text:
            score += 10.0
        if score > best.confidence:
            best = OCRResult(text=res.text, confidence=score, variant=f"{name} | {cfg}")

    return best


# -------------------------
# Cropping helpers
# -------------------------

def crop_rel(img_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Crop using relative coords [0..1]."""
    h, w = img_bgr.shape[:2]
    xa = max(0, min(w - 1, int(x1 * w)))
    xb = max(0, min(w, int(x2 * w)))
    ya = max(0, min(h - 1, int(y1 * h)))
    yb = max(0, min(h, int(y2 * h)))
    if xb <= xa or yb <= ya:
        return img_bgr[0:1, 0:1].copy()
    return img_bgr[ya:yb, xa:xb].copy()


def save_debug(debug_dir: Optional[str], name: str, img: np.ndarray) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)


# -------------------------
# Parsing helpers
# -------------------------

def find_ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", s or "")]


def parse_stage(text: str) -> Optional[str]:
    m = re.search(r"\b(\d-\d)\b", text.replace("—", "-").replace("–", "-"))
    return m.group(1) if m else None


def parse_unitcap(text: str) -> Optional[str]:
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", text)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def parse_gold_from_ocr(text: str, stage: Optional[str]) -> Optional[int]:
    ints = find_ints(text)
    if not ints:
        return None

    early = False
    if stage:
        try:
            a, _b = stage.split("-")
            early = int(a) <= 2
        except Exception:
            pass

    if early:
        small = [v for v in ints if 0 <= v <= 80]
        if small:
            return small[0]
        for v in ints:
            if v >= 100 and len(str(v)) == 3:
                first2 = int(str(v)[:2])
                if first2 <= 80:
                    return first2

    for v in ints:
        if 0 <= v <= 999:
            return v
    return None


def parse_synergies(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(.+?)\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            name = m.group(1).strip(" -•\t")
            cur = int(m.group(2))
            req = int(m.group(3))
            out.append({"name": name, "count": cur, "required": req, "raw": line})
    return out


# -------------------------
# Champion detection (optional helper)
# -------------------------

def detect_champion_from_image(
    image_path: str,
    champions: List[str],
    *,
    lang: str = "eng",
    tesseract_path: Optional[str] = None,
    fuzzy_threshold: int = 85,
) -> Optional[str]:
    def canon(s: str) -> str:
        s = s.lower().replace("_", " ")
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--oem 3 --psm 6"
    ocr = pytesseract.image_to_string(thr, lang=lang, config=config)
    ocr_c = canon(ocr)

    direct = []
    for name in champions:
        if canon(name) in ocr_c:
            direct.append(name)
    if direct:
        direct.sort(key=lambda n: len(canon(n)), reverse=True)
        return direct[0]

    best_name, best_score = None, 0
    for name in champions:
        score = fuzz.partial_ratio(canon(name), ocr_c)
        if score > best_score:
            best_name, best_score = name, score

    if best_name and best_score >= fuzzy_threshold:
        return best_name

    return None


# -------------------------
# Main extraction
# -------------------------

def extract_all(img_path: str, debug_dir: Optional[str] = None) -> Dict[str, Any]:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Impossible de lire l'image: {img_path}")

    h, w = img_bgr.shape[:2]

    rois = {
        "stage":        (0.33, 0.00, 0.46, 0.10),
        "gold":         (0.4771, 0.8157, 0.5255, 0.8435),
        "ping_fps":     (0.88, 0.00, 1.00, 0.12),

        "phase":        (0.40, 0.10, 0.60, 0.22),
        "unitcap":      (0.42, 0.16, 0.58, 0.34),

        "synergies":    (0.00, 0.12, 0.16, 0.60),

        "player_hp":    (0.8995, 0.1667, 0.9984, 0.7222),
        "player_level": (0.1427, 0.8157, 0.1792, 0.8407),

        "shop_row":     (0.246, 0.852, 0.774, 1.0),
    }

    raw_ocr: Dict[str, str] = {}
    conf: Dict[str, float] = {}

    stage_roi = crop_rel(img_bgr, *rois["stage"])
    save_debug(debug_dir, "roi_stage", stage_roi)
    stage_ocr = best_ocr(stage_roi, psm=11)
    raw_ocr["stage"] = stage_ocr.text
    conf["stage"] = stage_ocr.confidence
    stage = parse_stage(stage_ocr.text)

    gold_roi = crop_rel(img_bgr, *rois["gold"])
    save_debug(debug_dir, "roi_gold", gold_roi)
    gold_ocr = best_ocr(gold_roi, psm=6)
    raw_ocr["gold"] = gold_ocr.text
    conf["gold"] = gold_ocr.confidence
    gold = parse_gold_from_ocr(gold_ocr.text, stage)

    pf_roi = crop_rel(img_bgr, *rois["ping_fps"])
    save_debug(debug_dir, "roi_ping_fps", pf_roi)
    ping_ocr = best_ocr(pf_roi, psm=6)
    raw_ocr["ping_fps"] = ping_ocr.text
    conf["ping_fps"] = ping_ocr.confidence

    ping_ms = None
    fps = None
    m_ping = re.search(r"(\d+)\s*ms", ping_ocr.text.lower())
    if m_ping:
        ping_ms = int(m_ping.group(1))
    m_fps = re.search(r"fps\D*(\d+)", ping_ocr.text.lower())
    if m_fps:
        fps = int(m_fps.group(1))

    nums_pf = find_ints(ping_ocr.text)
    if ping_ms is None and nums_pf:
        ping_ms = nums_pf[0]
    if fps is None and len(nums_pf) >= 2:
        fps = nums_pf[-1]

    phase_roi = crop_rel(img_bgr, *rois["phase"])
    save_debug(debug_dir, "roi_phase", phase_roi)
    phase_ocr = best_ocr(phase_roi, psm=11)
    raw_ocr["phase"] = phase_ocr.text
    conf["phase"] = phase_ocr.confidence
    phase_guess = phase_ocr.text.strip() or None

    unitcap_roi = crop_rel(img_bgr, *rois["unitcap"])
    save_debug(debug_dir, "roi_unitcap", unitcap_roi)
    unitcap_ocr = best_ocr(unitcap_roi, psm=7, whitelist="0123456789/")
    raw_ocr["unitcap"] = unitcap_ocr.text
    conf["unitcap"] = unitcap_ocr.confidence
    unit_cap = parse_unitcap(unitcap_ocr.text)

    syn_roi = crop_rel(img_bgr, *rois["synergies"])
    save_debug(debug_dir, "roi_synergies", syn_roi)
    syn_ocr = best_ocr(syn_roi, psm=6)
    raw_ocr["synergies"] = syn_ocr.text
    conf["synergies"] = syn_ocr.confidence
    synergies = parse_synergies(syn_ocr.text)

    hp_roi = crop_rel(img_bgr, *rois["player_hp"])
    save_debug(debug_dir, "roi_player_hp", hp_roi)
    hp_ocr = best_ocr(hp_roi, psm=7, whitelist="0123456789")
    raw_ocr["player_hp"] = hp_ocr.text
    conf["player_hp"] = hp_ocr.confidence
    hp_ints = find_ints(hp_ocr.text)
    player_hp = hp_ints[0] if hp_ints else None

    lvl_roi = crop_rel(img_bgr, *rois["player_level"])
    save_debug(debug_dir, "roi_player_level", lvl_roi)
    lvl_ocr = best_ocr(lvl_roi, psm=6)
    raw_ocr["player_level"] = lvl_ocr.text
    conf["player_level"] = lvl_ocr.confidence
    lvl_ints = find_ints(lvl_ocr.text)
    player_level = lvl_ints[0] if lvl_ints else None

    shop_row = crop_rel(img_bgr, *rois["shop_row"])
    save_debug(debug_dir, "roi_shop_row", shop_row)

    shop: List[Dict[str, Any]] = []
    row_h, row_w = shop_row.shape[:2]
    slot_w = row_w / 5.0

    for i in range(5):
        x1 = int(i * slot_w)
        x2 = int((i + 1) * slot_w)
        card = shop_row[:, x1:x2].copy()
        save_debug(debug_dir, f"shop_card_{i+1}", card)

    return {
        "image": {"path": img_path, "width": w, "height": h},
        "raw_ocr": raw_ocr,
        "confidence": conf,
        "stage": stage,
        "gold": gold,
        "ping_ms": ping_ms,
        "fps": fps,
        "phase_guess": phase_guess,
        "unit_cap": unit_cap,
        "synergies": synergies,
        "player_hp": player_hp,
        "player_level": player_level,
        "shop": shop,
    }


# -------------------------
# CLI
# -------------------------
import cv2
import numpy as np
from typing import Union

def extract_largest_circle(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Prend une image (chemin fichier ou np.ndarray), détecte les cercles,
    sélectionne le plus grand, et retourne un crop BGRA (alpha transparent
    hors du cercle) contenant uniquement la zone du cercle le plus grand.
    """
    # --- Load image ---
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image}")
    else:
        img = image.copy()

    # Normalize channels to BGR for processing
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # --- HoughCircles (robuste sur UI) ---
    gray_blur = cv2.medianBlur(gray, 5)

    min_dist = max(20, int(min(h, w) * 0.12))
    min_r = max(10, int(min(h, w) * 0.08))
    max_r = int(min(h, w) * 0.60)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=120,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r
    )

    # Filtre simple pour éviter les faux cercles trop à gauche (optionnel mais utile en HUD)
    best = None  # (r, x, y)
    if circles is not None:
        circles = np.squeeze(circles).astype(np.float32)
        for x, y, r in circles:
            # Écarte les cercles coupés par les bords (souvent faux/partiels)
            if x - r < 0 or y - r < 0 or x + r > w or y + r > h:
                continue
            # HUD à droite : on ignore l'extrême gauche (ajuste si besoin)
            if x < w * 0.20:
                continue
            if best is None or r > best[0]:
                best = (r, x, y)

    # --- Fallback contours si Hough échoue ---
    if best is None:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < min_r or r > max_r:
                continue
            if x - r < 0 or y - r < 0 or x + r > w or y + r > h:
                continue
            if x < w * 0.20:
                continue
            if best is None or r > best[0]:
                best = (r, x, y)

    if best is None:
        raise RuntimeError("Aucun cercle détecté (Hough + fallback).")

    r, cx, cy = best
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))

    # --- Crop carré autour du cercle ---
    pad = 2
    x0 = max(0, cx - r - pad)
    y0 = max(0, cy - r - pad)
    x1 = min(w, cx + r + pad)
    y1 = min(h, cy + r + pad)
    crop_bgr = bgr[y0:y1, x0:x1]

    # --- Masque circulaire + alpha ---
    ch, cw = crop_bgr.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, thickness=-1)

    # Convert crop to BGRA and set alpha from mask
    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = mask

    return crop_bgra

def main():
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to TFT screenshot (png/jpg).")
    ap.add_argument("--out", default="tft_extract.json", help="Output JSON path.")
    ap.add_argument("--debug_dir", default=None, help="If set, saves ROI crops to this directory.")
    args = ap.parse_args()

    data = extract_all(args.image, debug_dir=args.debug_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    out = extract_largest_circle("./data/roi_player_level.png")
    cv2.imwrite("largest_circle.png", out)  # PNG gardera la transparence



if __name__ == "__main__":
    main()
