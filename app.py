import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["TORCH_HOME"] = "/tmp/torch"

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
# ESTILO (MODO OSCURO REAL)
# -----------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3, h4 {
    color: white;
}
.block-container {
    padding-top: 2rem;
}

/* Cards */
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
}

/* Métricas */
.metric {
    background-color: #22c55e;
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Texto secundario */
.subtext {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("⚙️ Panel")
st.sidebar.markdown("""
Sistema de detección de EPP

• Personas  
• Casco  
• Chaleco  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Mateo Sandoval")

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
# HEADER
# -----------------------
st.markdown("<h1>🏭 Detección de EPP</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Sistema de visión artificial para seguridad industrial</p>", unsafe_allow_html=True)

# -----------------------
# ENTRADA (UPLOAD + CÁMARA)
# -----------------------
st.markdown("### 📤 Subir imagen o tomar foto")

col_upload, col_camera = st.columns(2)

with col_upload:
    foto = st.file_uploader("Subir archivo", type=["jpg", "png", "jpeg"])

with col_camera:
    foto_camara = st.camera_input("Tomar foto")

# Si se toma foto, tiene prioridad
if foto_camara is not None:
    foto = foto_camara

# -----------------------
# PROCESAMIENTO
# -----------------------
if foto:
    imagen_original = Image.open(foto).convert("RGB")

    colA, colB = st.columns([2, 1])

    with colA:
        st.image(imagen_original, caption="Imagen cargada")

    img_np = np.array(imagen_original)
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    # -----------------------
    # MÉTRICA
    # -----------------------
    with colB:
        st.markdown(f"""
        <div class='metric'>
        {len(personas)}<br>Personas detectadas
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
            draw.text((x1o, max(0, y1o - 15)), label_espanol, fill="#22c55e")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Trabajador {i}")

        with col2:
            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("Cumple con EPP obligatorio")
            else:
                faltantes = requeridos - presentes
                st.error(f"Faltan: {', '.join(faltantes)}")

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
Desarrollado por Mateo Sandoval • 2026
</p>
""", unsafe_allow_html=True)
