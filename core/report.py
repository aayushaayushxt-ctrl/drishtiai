import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

def make_pdf(p: dict, orig, ov) -> bytes:
    buf = io.BytesIO(); s = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="DrishtiAI Screening Report")
    def im(x):
        b = io.BytesIO(); x.copy().convert("RGB").resize((240, 240)).save(b, "PNG"); b.seek(0); return RLImage(b, 200, 200)
    t = Table([["Patient", p["name"]], ["Patient ID", p["pid"]], ["Date", p["date"]],
               ["Image quality", f'{p["q"]}/100'], ["AI screening result", p["result"]],
               ["Model confidence", f'{p["conf"]:.0f}% (model confidence, not certainty)'],
               ["Output source", p["source"]]], colWidths=[130, 330])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E6F4F5"))]))
    story = [Paragraph("DrishtiAI — AI-Assisted Diabetic Eye Screening Report", s["Title"]), t, Spacer(1, 12),
             Table([[im(orig), im(ov)]]), Paragraph("Original image &nbsp;&nbsp;|&nbsp;&nbsp; Grad-CAM overlay (areas that influenced the model)", s["Italic"]),
             Spacer(1, 10), Paragraph("<b>Recommendation:</b> " + p["rec"], s["Normal"]), Spacer(1, 10),
             Paragraph("<b>Disclaimer:</b> This is an AI-assisted screening result from a research prototype. It does not constitute a medical diagnosis and requires professional review. Highlighted regions show what influenced the model, not confirmed lesions.", s["Normal"])]
    if p["source"] != "real_model": story.append(Paragraph("<b>MOCK OUTPUT — NOT FOR CLINICAL USE</b>", s["Normal"]))
    doc.build(story); return buf.getvalue()
