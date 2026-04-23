import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# -----------------------
# CONFIGURACIÓN
# -----------------------
st.set_page_config(page_title="Detección PPE", layout="wide")

# -----------------------
# ESTILO GLOBAL (MEJOR CONTRASTE)
# -----------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3, h4, h5 {
    color: #111827;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}
.metric {
    background-color: #1e293b;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR (NUEVA ESTRUCTURA)
# -----------------------
st.sidebar.title("⚙️ Panel")
st.sidebar.markdown("""
**Sistema de detección PPE**

Sube una imagen y analiza:
- Personas detectadas
- Uso de casco
- Uso de chaleco
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Mateo Sandoval**")

# -----------------------
# MODELOS
# -----------------------
TRADUCCION_CLASES = {
    "boots": "Botas",
    "earmuffs": "Orejeras",
    "glasses": "Gafas",
    "gloves": "Guantes",
    "helmet": "Casco",
    "person": "Persona",
    "vest": "Chaleco"
}

@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# HEADER DIFERENTE (MINIMALISTA)
# -----------------------
st.markdown("""
<h1 style='margin-bottom:0;'>🏭 Detección Inteligente de EPP</h1>
<p style='color:gray; margin-top:0;'>Sistema de visión artificial para seguridad industrial</p>
""", unsafe_allow_html=True)

# -----------------------
# UPLOAD
# -----------------------
st.markdown("### 📤 Subir imagen")
foto = st.file_uploader("", type=["jpg", "png", "jpeg"])

# -----------------------
# PROCESAMIENTO
# -----------------------
if foto:
    imagen_original = Image.open(foto).convert("RGB")

    colA, colB = st.columns([2, 1])

    with colA:
        st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    img_np = np.array(imagen_original)
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    # -----------------------
    # MÉTRICA PRINCIPAL
    # -----------------------
    with colB:
        st.markdown(f"""
        <div class='metric'>
        <h2>{len(personas)}</h2>
        <p>Personas detectadas</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------
    # TRABAJADORES
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):

        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)
        resultados_ppe = modelo_ppe(persona_np)[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []
        datos_analitica = []

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_ppe.names[cls]

            if label_ingles == "person":
                continue

            label_espanol = TRADUCCION_CLASES.get(label_ingles, label_ingles.capitalize())
            conf = float(box.conf[0])

            etiquetas.append(label_espanol)
            datos_analitica.append({
                "Equipo": label_espanol,
                "Confianza": f"{conf*100:.1f}%"
            })

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#22c55e", width=3)
            draw.text((x1o, max(0, y1o - 15)), f"{label_espanol}", fill="#22c55e")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Trabajador {i}", use_container_width=True)

        with col2:
            # -----------------------
            # ESTADO
            # -----------------------
            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("🟢 Cumple con EPP obligatorio")
            else:
                faltantes = requeridos - presentes
                st.error(f"🔴 Faltan: {', '.join(faltantes)}")

            # -----------------------
            # TABLA
            # -----------------------
            if datos_analitica:
                df = pd.DataFrame(datos_analitica)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Sin detecciones")

        st.markdown("---")

# -----------------------
# FOOTER
# -----------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Desarrollado por <b>Mateo Sandoval</b> • 2026
</p>
""", unsafe_allow_html=True)
