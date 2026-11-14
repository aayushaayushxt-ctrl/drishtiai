"""DrishtiAI — Explainable AI for Diabetic Retinopathy Screening in Rural India (research prototype)."""
import io, datetime as dt
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from PIL import Image
from core.i18n import LANGS, t
from core import model_api as M, quality, report

st.set_page_config(page_title="DrishtiAI", page_icon="👁️", layout="wide")
ss = st.session_state
for k, v in dict(auth=False, role="Patient", lang="en", rural=False, big=False, contrast=False, step=1, img=None,
                 res=None, q=None, page="dashboard", demo=True,
                 hist=[("12 Nov 2026", "Healthy", 93, 88), ("18 Aug 2026", "Healthy", 91, 84), ("10 May 2026", "Diseased", 82, 79)]).items():
    ss.setdefault(k, v)
L = lambda k: t(k, ss.lang)
fs = "1.2rem" if (ss.big or ss.rural) else "1rem"
bg, fg = ("#000", "#fff") if ss.contrast else ("#F4F8FB", "#12263F")
st.markdown(f"""<style>
html, body, .stApp, p, li, label, span {{font-size:{fs};}}
.stApp {{background:{bg}; color:{fg};}}
.hero {{background:linear-gradient(120deg,#0B2E5C 0%,#0E7C86 60%,#2FB38B 100%);color:#fff;padding:28px 32px;border-radius:20px;margin-bottom:18px}}
.hero h1,.hero p{{color:#fff;margin:0}} .hero p{{opacity:.9;margin-top:6px}}
.card{{background:#fff;color:#12263F;border-radius:16px;padding:16px 18px;border:1px solid #DCE7EE;box-shadow:0 2px 10px rgba(14,60,100,.06);transition:transform .2s}}
.card:hover{{transform:translateY(-3px)}} .card b{{font-size:1.5rem}} .card small{{color:#5B7088}}
.ai{{border-left:6px solid #5B4BDB}} .ok{{border-left:6px solid #1E9E6A}} .warn{{border-left:6px solid #E8890C}}
.banner{{background:#FFF4DC;color:#7A4B00;border:1px dashed #E8890C;padding:6px 12px;border-radius:10px;font-weight:600;margin-bottom:10px}}
.disc{{background:#EEF1FF;color:#2B2A6B;padding:10px 14px;border-radius:12px;font-size:.9rem}}
.stButton>button{{border-radius:12px;min-height:{'3.4rem' if ss.rural else '2.6rem'};font-weight:600}}
{'@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}' }
{'*{animation:none!important;transition:none!important}' if ss.rural else ''}
</style>""", unsafe_allow_html=True)

LOGO = "<svg width='38' height='38' viewBox='0 0 40 40'><circle cx='20' cy='20' r='19' fill='#0B2E5C'/><path d='M4 20C10 10 30 10 36 20 30 30 10 30 4 20Z' fill='none' stroke='#7FE3D0' stroke-width='2.4'/><circle cx='20' cy='20' r='6' fill='#5B4BDB'/><circle cx='20' cy='20' r='2.4' fill='#fff'/><path d='M29 9l1.5 3 3 1.5-3 1.5L29 18l-1.5-3-3-1.5 3-1.5z' fill='#FFC857'/></svg>"
def card(title, val, sub="", cls=""):
    st.markdown(f"<div class='card {cls}'><small>{title}</small><br><b>{val}</b><br><small>{sub}</small></div>", unsafe_allow_html=True)
def disc(): st.markdown(f"<div class='disc'>ℹ️ {L('disclaimer')}</div>", unsafe_allow_html=True)
def demo_banner():
    if ss.demo: st.markdown("<div class='banner'>DEMO DATA — NOT FOR CLINICAL USE</div>", unsafe_allow_html=True)

# ---------------- LOGIN / ONBOARDING ----------------
if not ss.auth:
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown(f"<div class='hero'><div style='display:flex;gap:12px;align-items:center'>{LOGO}<h1>DrishtiAI</h1></div><p>{L('tagline')}</p><p style='margin-top:18px'>👁️ AI-assisted screening · 🔍 Explainable (Grad-CAM) · 🌾 Built for rural care · 🌐 8 Indian languages</p></div>", unsafe_allow_html=True)
        ss.lang = st.selectbox("🌐 Language / भाषा", list(LANGS), format_func=LANGS.get, index=list(LANGS).index(ss.lang))
        disc()
    with b:
        tab1, tab2 = st.tabs(["📱 Mobile + OTP", "📝 Register (4 steps)"])
        with tab1:
            c1, c2 = st.columns([1, 3]); c1.selectbox("Code", ["+91", "+880", "+977", "+94"]); ph = c2.text_input("Mobile number", "9876543210")
            ss.role = st.radio("Role", ["Patient", "Community Health Worker", "Researcher / Admin"], horizontal=True)
            if st.button("Send OTP", use_container_width=True): ss.otp_sent = True; st.toast("OTP sent (prototype: use 123456)")
            if ss.get("otp_sent"):
                otp = st.text_input("Enter 6-digit OTP", max_chars=6, type="password"); st.caption("Resend in 30s · Prototype auth — no real SMS/security")
                if st.button("Verify & Login", type="primary", use_container_width=True):
                    if otp == "123456": ss.auth = True; st.rerun()
                    else: st.error("Incorrect OTP. Prototype OTP is 123456.")
            st.divider()
            if st.button("🎓 Professor / Demo Mode — skip login", use_container_width=True): ss.auth = True; ss.demo = True; st.rerun()
        with tab2:
            n = ss.get("reg", 1); st.progress(n / 4, text=["Basic information", "Diabetes information", "Eye health", "Consent & privacy"][n - 1])
            if n == 1: st.text_input("Full name"); st.number_input("Age", 1, 110, 45); st.selectbox("Gender", ["Female", "Male", "Other"]); st.text_input("Village / district")
            if n == 2: st.selectbox("Diabetes status", ["Diagnosed", "Pre-diabetic", "Not sure"]); st.selectbox("Type", ["Type 2", "Type 1", "Unknown"]); st.number_input("Years since diagnosis", 0, 60, 5)
            if n == 3: st.multiselect("Known eye problems", ["Blurred vision", "Floaters", "Cataract", "Glaucoma", "None"]); st.checkbox("Eye exam in the last 12 months")
            if n == 4: st.checkbox("I understand AI screening is not a diagnosis and I consent to my image being used for screening."); st.caption("Only minimum data is collected. Prototype — no real storage.")
            c1, c2 = st.columns(2)
            if n > 1 and c1.button("← Back"): ss.reg = n - 1; st.rerun()
            if n < 4 and c2.button("Next →", type="primary"): ss.reg = n + 1; st.rerun()
            if n == 4 and c2.button("Finish → Dashboard", type="primary"): ss.auth = True; st.rerun()
    st.stop()

# ---------------- SIDEBAR ----------------
nav = ["dashboard", "new_screening", "history", "learn", "research", "settings"]
if ss.role != "Patient": nav.insert(2, "patients")
if ss.role == "Researcher / Admin": nav.insert(5, "analytics")
names = {k: L(k) for k in nav}; names.update(patients="Patients & Queue", analytics="Analytics")
with st.sidebar:
    st.markdown(f"<div style='display:flex;gap:10px;align-items:center'>{LOGO}<b style='font-size:1.4rem'>DrishtiAI</b></div>", unsafe_allow_html=True)
    ss.rural = st.toggle("🌾 Rural Care Mode", ss.rural, help="Larger controls, fewer animations, simple language")
    ss.page = st.radio("Menu", nav, format_func=names.get, index=nav.index(ss.page) if ss.page in nav else 0, label_visibility="collapsed")
    ss.lang = st.selectbox("🌐", list(LANGS), format_func=LANGS.get, index=list(LANGS).index(ss.lang))
    ss.net = st.selectbox("📶 Connectivity (simulated)", ["Online", "Limited", "Offline"])
    if ss.net == "Offline": st.warning("Your screening data will sync when connectivity is restored.")
    ss.demo = st.toggle("Demo mode", ss.demo)
    st.caption(f"Role: {ss.role} · session active")
    if st.button("Log out"): ss.auth = False; ss.otp_sent = False; st.rerun()
P = ss.page
demo_banner()

# ---------------- PAGES ----------------
if P == "dashboard":
    hr = dt.datetime.now().hour; g = "Good morning" if hr < 12 else "Good afternoon" if hr < 18 else "Good evening"
    who = {"Patient": "Aayush", "Community Health Worker": "Sunita (ASHA)", "Researcher / Admin": "Researcher"}[ss.role]
    st.markdown(f"<div class='hero'><h1>{g}, {who} 👋</h1><p>{L('greeting')} — {L('tagline')}</p></div>", unsafe_allow_html=True)
    if ss.role == "Patient":
        h = ss.hist[0]; c = st.columns(4)
        with c[0]: card("Last screening", h[0], "AI-assisted", "ai")
        with c[1]: card("Previous result", "Low concern" if h[1] == "Healthy" else "Review advised", f"model confidence {h[2]}%", "ok" if h[1] == "Healthy" else "warn")
        with c[2]: card("Follow-up due", "12 Nov 2027" if h[1] == "Healthy" else "Within 4 weeks", "demo schedule")
        with c[3]: card("Diabetes duration", "6 years", "Type 2 (demo)")
    else:
        c = st.columns(4)
        for col, (a, b) in zip(c, [("Patients screened", "128"), ("Pending reviews", "9"), ("Follow-ups due", "14"), ("Today's screenings", "6")]):
            with col: card(a, b, "demo number", "ai")
    st.write("")
    q = st.columns(3)
    if q[0].button("▶ " + L("start"), type="primary", use_container_width=True): ss.page = "new_screening"; ss.step = 1; st.rerun()
    if q[1].button("📄 View previous reports", use_container_width=True): ss.page = "history"; st.rerun()
    if q[2].button("📚 " + L("learn"), use_container_width=True): ss.page = "learn"; st.rerun()
    st.subheader("Health timeline")
    for d, lab, cf, qs in ss.hist: st.markdown(f"<div class='card {'ok' if lab=='Healthy' else 'warn'}'><b style='font-size:1.1rem'>{d}</b> — {'No obvious signs' if lab=='Healthy' else 'Possible signs — review advised'} · model confidence {cf}% · image quality {qs}/100</div>", unsafe_allow_html=True)
    disc()

elif P == "new_screening":
    labels = ["Confirm patient", "Capture / upload", "Quality check", "AI analysis", "Result & explanation", "Report"]
    st.progress(ss.step / 6, text=f"Step {ss.step}/6 — {labels[ss.step-1]}")
    if ss.step == 1:
        c = st.columns(2)
        with c[0]: card("Patient", "Aayush Kumar", "ID DRI-2026-0142 · Age 52 · Type 2 diabetes, 6 yrs (demo)", "ai")
        with c[1]: st.markdown("<div class='card'><b style='font-size:1.1rem'>Consent</b><br><small>Your image is used only to produce an AI-assisted screening result. It is not a diagnosis; professional review may be needed.</small></div>", unsafe_allow_html=True)
        ok = st.checkbox("I understand and consent")
        if st.button("Continue →", type="primary", disabled=not ok): ss.step = 2; st.rerun()
    elif ss.step == 2:
        st.markdown("**Tips:** dark room · pupil dilated if possible · centre the optic disc · keep steady. ❌ Blurry ❌ Too dark ❌ Off-centre")
        t1, t2, t3 = st.tabs(["⬆ Upload fundus image", "📷 Camera (prototype)", "🧪 Demo sample"])
        with t1: up = st.file_uploader("PNG / JPG, up to 15 MB", type=["png", "jpg", "jpeg"])
        with t2: cam = st.camera_input("Capture")
        with t3: smp = st.selectbox("Synthetic sample (not a real patient image)", [1, 2, 3], format_func=lambda i: f"Synthetic sample {i}")
        src = up or cam
        if src: img = Image.open(src).convert("RGB")
        elif st.button("Use synthetic sample"): img = M.demo_fundus(smp)
        else: img = None
        if ss.rural and img is not None: img.thumbnail((1024, 1024)); st.caption("Low-bandwidth: image compressed before analysis.")
        if img is not None: ss.img = img; ss.q = quality.assess(img); ss.step = 3; st.rerun()
    elif ss.step == 3:
        q = ss.q; c = st.columns([1, 1])
        c[0].image(ss.img, use_container_width=True)
        with c[1]:
            card("Image quality", f"{q['score']}/100 — {q['label']}", "Computed from focus, brightness, contrast, field of view, resolution", "ok" if q["score"] >= 50 else "warn")
            for k, v in q["parts"].items(): st.progress(v / 100, text=f"{k}: {v}")
        if q["label"] == "Poor": st.error("Quality is poor — please capture another image."); 
        a, b = st.columns(2)
        if a.button("↺ Retake / re-upload"): ss.step = 2; st.rerun()
        if b.button("Run AI analysis →", type="primary", disabled=q["label"] == "Poor" and not ss.demo): ss.step = 4; st.rerun()
    elif ss.step == 4:
        bar = st.progress(0, text="Starting…")
        for i, s in enumerate(["Image preprocessing", "Retinal feature extraction", "AI model inference", "Explainability (Grad-CAM)", "Risk assessment"]):
            bar.progress((i + 1) / 5, text=s)
            if i == 2: ss.res = M.analyze(ss.img)
        ss.step = 5; st.rerun()
    elif ss.step == 5:
        r = ss.res; flag = r["label"] == "Diseased"; low = r["confidence"] < 70
        if r["source"] != "real_model": st.markdown(f"<div class='banner'>MOCK OUTPUT — real model not loaded. {r.get('note','')}</div>", unsafe_allow_html=True)
        c = st.columns([1.2, 1])
        with c[0]: card("Screening result", L("result_flag") if flag else L("result_ok"), "Uncertain — confidence is low; please retake or seek review" if low else "AI-assisted · needs professional review", "warn" if flag or low else "ok")
        with c[1]: card(L("confidence"), f"{r['confidence']:.0f}%", "model confidence, not certainty", "ai")
        st.subheader("🔍 " + L("why"))
        mode = st.radio("View", ["Overlay", "Original", "Heatmap only", "Side by side"], horizontal=True)
        op = st.slider("Heatmap opacity", 0.0, 1.0, 0.65, 0.05); zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.25)
        ov = M.overlay(ss.img, r["heatmap"], op); w, h = ss.img.size
        def zm(im):
            if zoom == 1: return im
            cw, ch = int(w / zoom), int(h / zoom); return im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
        if mode == "Side by side": a, b = st.columns(2); a.image(zm(ss.img), caption="Original"); b.image(zm(ov), caption="Grad-CAM overlay")
        else: st.image(zm({"Overlay": ov, "Original": ss.img, "Heatmap only": M.overlay(Image.new("RGB", ss.img.size), r["heatmap"], 1)}[mode]), use_container_width=True)
        st.caption("Brighter colours = areas that influenced the model's output. These are NOT confirmed lesions.")
        hm = r["heatmap"]; top = np.dstack(np.unravel_index(np.argsort(hm.ravel())[::-1][:3000], hm.shape))[0]
        cy, cx = top.mean(0); pos = ("upper" if cy < h / 2 else "lower") + "-" + ("left" if cx < w / 2 else "right")
        st.info(f"The model gave most weight to the **{pos}** part of the image ({100*(hm>0.5).mean():.0f}% of pixels above half peak importance). Grad-CAM shows where the network looked for its learned features; it does not prove those regions are disease.")
        st.subheader("Next steps")
        st.success("Routine yearly screening advised (diabetes)") if not flag and not low else st.warning("Professional review recommended — contact your health worker or eye-care facility (demo facility list on Follow-ups).")
        disc()
        if st.button("Create report →", type="primary"):
            ss.hist.insert(0, (dt.date.today().strftime("%d %b %Y"), r["label"], int(r["confidence"]), ss.q["score"])); ss.step = 6; st.rerun()
    elif ss.step == 6:
        r, q = ss.res, ss.q; flag = r["label"] == "Diseased"
        p = dict(name="Aayush Kumar", pid="DRI-2026-0142", date=dt.date.today().strftime("%d %b %Y"), q=q["score"], result=L("result_flag") if flag else L("result_ok"),
                 conf=r["confidence"], source=r["source"], rec="Professional review recommended." if flag else "Routine yearly screening.")
        ov = M.overlay(ss.img, r["heatmap"], .65)
        st.success("Screening saved to history.")
        st.download_button("⬇ Download PDF report", report.make_pdf(p, ss.img, ov), "DrishtiAI_report.pdf", "application/pdf", type="primary")
        st.button("Share with healthcare professional (prototype)", on_click=lambda: st.toast("Prototype: sharing not implemented"))
        if st.button("Start another"): ss.step = 1; st.rerun()

elif P == "history":
    st.title("📚 " + L("history")); f = st.selectbox("Filter", ["All", "Healthy", "Diseased"])
    for d, lab, cf, qs in [x for x in ss.hist if f in ("All", x[1])]:
        with st.expander(f"{d} — {'No obvious signs' if lab=='Healthy' else 'Possible signs'} · {cf}%"):
            st.write(f"Model confidence {cf}% · Image quality {qs}/100 · Requires professional review for any concern."); st.caption("Demo history entry.")
    if not ss.hist: st.info("No screenings yet.")

elif P == "patients":
    st.title("Patients & screening queue"); demo_banner()
    df = pd.DataFrame({"ID": [f"DRI-2026-{i:04d}" for i in range(140, 148)], "Name": ["Aayush K", "Meena D", "Raju S", "Lakshmi P", "Harpreet S", "Anita B", "Vijay M", "Fatima R"],
                       "Village": ["Hodal", "Palwal", "Hodal", "Kosi", "Palwal", "Hodal", "Kosi", "Palwal"], "Status": ["Screened", "Pending review", "Queued", "Screened", "Follow-up due", "Queued", "Screened", "Pending review"]})
    s = st.text_input("🔎 Search"); st.dataframe(df[df.apply(lambda r: s.lower() in " ".join(r).lower(), axis=1)], use_container_width=True, hide_index=True)
    st.button("➕ Register patient (creates ID)", on_click=lambda: st.toast("Prototype: DRI-2026-0148 created"))
    st.caption("Offline queue: records stay on device and sync when connected (simulated).")

elif P == "analytics":
    st.title("Researcher dashboard"); demo_banner(); st.warning("All figures are randomly generated DEMO data — not study results.")
    r = np.random.default_rng(3); m = pd.DataFrame({"Month": ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"], "Screenings": r.integers(20, 90, 6)})
    a, b = st.columns(2)
    a.plotly_chart(px.line(m, x="Month", y="Screenings", markers=True, title="Screening trend (demo)", color_discrete_sequence=["#0E7C86"]), use_container_width=True)
    b.plotly_chart(px.pie(values=[70, 30], names=["Healthy", "Possible signs"], hole=.5, title="Result distribution (demo)", color_discrete_sequence=["#1E9E6A", "#E8890C"]), use_container_width=True)
    a.plotly_chart(px.bar(x=list(LANGS.values()), y=r.integers(5, 40, 8), title="Language usage (demo)", color_discrete_sequence=["#5B4BDB"]), use_container_width=True)

elif P == "learn":
    st.title("📖 Learn about diabetic retinopathy")
    cs = [("What is it?", "High blood sugar can damage tiny blood vessels at the back of the eye."), ("Why screen?", "Early changes often cause no symptoms; screening can catch them in time."),
          ("During screening", "A quick, painless photo of the back of your eye."), ("What the AI does", "Looks at the photo and gives a screening result with a highlighted explanation."),
          ("What the AI cannot do", "It cannot diagnose, replace an eye doctor, or see everything.")]
    cols = st.columns(3)
    for i, (a, b) in enumerate(cs):
        with cols[i % 3]: card(a, "", b, "ai")
    st.subheader("Myth vs Fact")
    for m, f in [("If I see well, my eyes are fine.", "Diabetic eye changes can be silent. Regular screening matters."), ("AI can replace the doctor.", "No. AI only assists screening; a professional decides.")]:
        with st.expander("❓ " + m): st.write("✅ " + f)

elif P == "research":
    t1, t2, t3, t4 = st.tabs(["About", "Validation", "Roadmap", "Model API"])
    with t1: st.markdown("**Problem** — DR can cause preventable vision loss; specialist access is limited in rural areas.\n\n**Solution** — AI-assisted retinal screening with explainability.\n\n**Why XAI?** — So users and clinicians can see which regions influenced the model.\n\n**Objective** — Develop and evaluate an explainable screening prototype for resource-constrained settings.")
    with t2:
        st.info("To be populated with experimentally validated results.")
        st.table(pd.DataFrame({"Item": ["Dataset", "No. of images", "Architecture", "Preprocessing", "Train/test split", "Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1", "AUC", "Explainability method"],
                               "Value": ["—"] * 11 + ["Grad-CAM (implemented)"]}))
    with t3: st.markdown("Current prototype → Real clinical dataset → Model validation → Explainability evaluation → Healthcare-worker pilot → Field validation → Scalable deployment")
    with t4:
        st.json(M.MODEL_INFO); st.code('{"label":"Diseased","confidence":91.2,"image_quality":87,"explanation_available":true,"heatmap":"HxW float array","source":"real_model"}', "json")
        st.caption("Swap point: core/model_api.py → analyze(). Loads model/best_model.pth; falls back to labelled MOCK if torch is missing.")

elif P == "settings":
    st.title("⚙️ " + L("settings"))
    ss.lang = st.selectbox("Language", list(LANGS), format_func=LANGS.get, index=list(LANGS).index(ss.lang), key="s_l")
    ss.big = st.toggle("Large text", ss.big); ss.contrast = st.toggle("High contrast", ss.contrast)
    st.subheader("AI model information"); st.json(M.MODEL_INFO)
    st.caption("Prototype: no real security, notifications or storage implemented.")
