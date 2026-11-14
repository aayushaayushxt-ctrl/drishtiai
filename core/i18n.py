import json, pathlib
D = pathlib.Path(__file__).resolve().parent.parent / "locales"
LANGS = {"en": "English", "hi": "हिन्दी", "bn": "বাংলা", "mr": "मराठी", "ta": "தமிழ்", "te": "తెలుగు", "gu": "ગુજરાતી", "pa": "ਪੰਜਾਬੀ"}
_c = {}
def _load(l):
    if l not in _c: _c[l] = json.loads((D / f"{l}.json").read_text("utf-8"))
    return _c[l]
def t(key, lang):
    return _load(lang).get(key) or _load("en").get(key, key)
