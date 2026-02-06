# Prerequis
pip install opencv-python numpy pillow pytesseract rapidfuzz

**Tesseract OCR** a installer aussi


# EXECUTION
python readUI.py --image "vlcsnap-2026-01-27-22h58m13s582.png" --out "tft.json" --debug_dir "debug"

Extrait les sprite UI dans le dossier **debug** et essaye de lire ce qu'il peut dans **tft.json**

# Lecture des champions

python .\ocr_match_champion.py -q ".\debug\shop_card_1.png" --lang eng

lis le nom du champion depuis sa carte **shop_card_1.png** extraite de l'UI et retourne sa valeur
