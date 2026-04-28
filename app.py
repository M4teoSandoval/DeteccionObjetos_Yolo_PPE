import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["TORCH_HOME"] = "/tmp/torch"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# -----------------------
# CONFIGURACIÓN
# -----------------------
st.set_page_config(page_title="Detección PPE", layout="wide")

# -----------------------
# ESTILO
# -----------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
h1, h2, h3, h4 { color: white; }
.block-container { padding-top: 2rem; }

.metric {
    background-color: #22c55e;
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

.subtext { color: #94a3b8; }
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
    return YOLO("yolov8n.pt"), YOLO("best.pt")

modelo_personas, modelo_ppe = load_models()

# -----------------------
# HEADER
# -----------------------
st.markdown("<h1>🏭 Detección de EPP</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Sistema de visión artificial para seguridad industrial</p>", unsafe_allow_html=True)

# -----------------------
# ENTRADA (LIMPIA Y ORDENADA)
# -----------------------
st.markdown("### 📤 Subir imagen o tomar foto")

opcion = st.radio("Selecciona fuente:", ["Subir imagen", "Tomar foto"], horizontal=True)

foto = None

if opcion == "Subir imagen":
    foto = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])
else:
    foto = st.camera_input("Toma una foto")

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

    personas = [
        tuple(map(int, box.xyxy[0]))
        for box in resultados_personas.boxes
        if int(box.cls[0]) == 0
    ]

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
        resultados_ppe = modelo_ppe(np.array(persona_crop))[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label = modelo_ppe.names[cls]

            if label == "person":
                continue

            label_es = TRADUCCION_CLASES.get(label, label.capitalize())
            etiquetas.append(label_es)

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#22c55e", width=3)
            draw.text((x1o, max(0, y1o - 15)), label_es, fill="#22c55e")

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
