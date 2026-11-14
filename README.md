# DrishtiAI — Explainable AI for Diabetic Retinopathy Screening in Rural India
Research prototype (Streamlit, pure Python). AI-assisted screening only — **not a medical diagnosis**.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
Login: any mobile number, OTP `123456` (prototype), or click **Professor / Demo Mode**.

## Live link (free): Streamlit Community Cloud
1. Push this folder to a GitHub repo (model/best_model.pth is ~90 MB, under GitHub's 100 MB limit).
2. https://share.streamlit.io → New app → pick repo, branch, main file `app.py` → Deploy.
3. You get a public `https://<name>.streamlit.app` URL.

## Connect / replace the model
Everything goes through `core/model_api.py → analyze(image)`. It loads `model/best_model.pth`
via your `predict.py` + `grad_cam.py`. Your model is **binary (Healthy/Diseased, APTOS 2019)**, so the UI
shows two outcomes, not 5 DR grades. For a 5-grade model, retrain and extend the label mapping in app.py.
If torch is missing, the app shows a clearly labelled MOCK output.

## Structure
app.py (UI/pages) · core/{i18n,quality,model_api,report}.py · locales/*.json (8 languages) · model/ (your model)

## Honest status
Real: image-quality scoring, your model + Grad-CAM, PDF report, 8-language UI shell, roles, Rural Care Mode.
Mock / pending: OTP & auth, database/persistence, patients & analytics data, facilities, sharing, offline sync.
Validation page is intentionally empty until real results exist.
