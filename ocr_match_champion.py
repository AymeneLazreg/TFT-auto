import re
import argparse

import cv2
import pytesseract
from rapidfuzz import fuzz

def canon(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ✅ Mets ici TA liste exacte des 101 champions (orthographe identique au texte affiché ingame)
CHAMPIONS_101 = [
    # Exemples (remplace/complète avec tes 101 noms)
    "Aatrox", "Ahri", "Ambessa", "Anivia", "Annie", "Aphelios", "Ashe",
    "Aurelion Sol", "Azir", "Bard", "Baron Nashor", "Bel Veth", "Blitzcrank",
    "Braum", "Briar", "Brock", "Caitlyn", "Cho Gath", "Darius", "Diana",
    "Dr Mundo", "Draven", "Ekko", "Fiddlesticks", "Fizz", "Galio", "Gangplank",
    "Garen", "Graves", "Gwen", "Heraut de la Faille", "Illaoi", "Jarvan IV",
    "Jhin", "Jinx", "Kai Sa", "Kalista", "Kennen", "Kindred",
    "Kobuko et Yuumi", "Kog Maw", "LeBlanc", "Leona", "Lissandra", "Loris",
    "Lucian et Senna", "Lulu", "Lux", "Malzahar", "Mel", "Milio",
    "Miss Fortune", "Nasus", "Nautilus", "Neeko", "Nidalee", "Orianna",
    "Ornn", "Poppy", "Qiyana", "Rek Sai", "Renekton", "Rumble", "Ryze",
    "Sejuani", "Seraphine", "Sett", "Shen", "Shyvana", "Singed", "Sion",
    "Skarner", "Sona", "Swain", "Sylas", "T-Hex", "Tahm Kench", "Taric",
    "Teemo", "Thresh", "Tibbers", "Tristana", "Tryndamere", "Twisted Fate",
    "Vayne", "Veigar", "Vi", "Viego", "Volibear", "Warwick", "Wukong",
    "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Yunara", "Zaahen",
    "Ziggs", "Zilean", "Zoe",
]

def ocr_text(image_path: str, lang="eng"):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {image_path}")

    # Pré-traitement OCR (souvent utile pour UI)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(thr, lang=lang, config=config)

def find_best_champion(ocr: str, champions, fuzzy_threshold=85):
    ocr_c = canon(ocr)

    # 1) match direct (le plus fiable)
    direct = []
    for name in champions:
        if canon(name) in ocr_c:
            direct.append(name)
    if direct:
        direct.sort(key=lambda n: len(canon(n)), reverse=True)
        return direct[0], 100, "direct"

    # 2) fuzzy (si OCR a fait des fautes)
    best_name, best_score = None, 0
    for name in champions:
        score = fuzz.partial_ratio(canon(name), ocr_c)
        if score > best_score:
            best_name, best_score = name, score

    if best_name and best_score >= fuzzy_threshold:
        return best_name, best_score, "fuzzy"

    return None, best_score, "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required=True, help="Image requête")
    ap.add_argument("--lang", default="eng", help="Langue tesseract (eng conseillé)")
    ap.add_argument("--tesseract", default=None, help="Chemin tesseract.exe si pas dans PATH")
    ap.add_argument("--threshold", type=int, default=85, help="Seuil fuzzy (0-100)")
    args = ap.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    text = ocr_text(args.query, lang=args.lang)
    

    champ, score, mode = find_best_champion(text, CHAMPIONS_101, fuzzy_threshold=args.threshold)
    if champ:
        print(f"\nChampion détecté : {champ} | score={score} | mode={mode}\n")
    else:
        print(f"Aucun champion détecté. Meilleur score fuzzy: {score}")

if __name__ == "__main__":
    main()
