import os
# Evita el warning de Ultralytics en Streamlit Cloud
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# -----------------------
# Diccionario de Traducción
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

# -----------------------
# Configuración de página
# -----------------------
st.set_page_config(page_title="Detección de PPE", layout="wide")

# -----------------------
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div style='background: linear-gradient(90deg, #0f172a, #1e3a8a); padding: 25px; border-radius: 12px'>
    <h1 style='color: white; margin: 0;'>🏭 Sistema Inteligente de Detección de EPP</h1>
    <p style='color: #cbd5f5; margin: 5px 0 0 0;'>
        Monitoreo automático de seguridad industrial mediante visión artificial
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# INSTRUCCIONES
# -----------------------
st.markdown("""
<div style='background-color:#f1f5f9; padding:20px; border-radius:10px; margin-top:15px;'>
<b>📌 Instrucciones de uso:</b>
<ol>
<li>Sube una fotografía del trabajador.</li>
<li>El sistema detectará automáticamente a las personas.</li>
<li>Se verificará el uso de <b>Casco y Chaleco</b>.</li>
<li>El semáforo indicará el estado de acceso.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# -----------------------
# UPLOADER
# -----------------------
st.markdown("### 📤 Cargar imagen")
foto = st.file_uploader("", type=["jpg", "png", "jpeg"])

# -----------------------
# PROCESAMIENTO
# -----------------------
if foto:
    imagen_original = Image.open(foto).convert("RGB")
    
    with st.expander("🖼️ Ver imagen original", expanded=False):
        st.image(imagen_original, use_container_width=True)

    img_np = np.array(imagen_original)

    # Detectar personas
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    st.markdown(f"""
    <div style='background-color:#e0f2fe; padding:15px; border-radius:10px; margin-top:10px;'>
    <h3 style='margin:0;'>👥 Personas detectadas: {len(personas)}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Procesar cada persona
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):

        st.markdown(f"""
        <div style='background-color:#111827; padding:10px; border-radius:8px; margin-top:10px;'>
        <h4 style='color:white; margin:0;'>👤 Trabajador {i}</h4>
        </div>
        """, unsafe_allow_html=True)

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
                "Equipo Detectado": label_espanol,
                "Confianza": f"{conf*100:.2f}%"
            })

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#00FF00", width=3)
            draw.text((x1o, max(0, y1o - 15)), f"{label_espanol} {conf:.2f}", fill="#00FF00")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Trabajador {i}", use_container_width=True)

        with col2:
            st.markdown("#### 🚥 Control de Acceso")

            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.markdown("""
                <div style='background-color:#dcfce7; padding:15px; border-radius:10px;'>
                🟢 <b>ACCESO PERMITIDO</b><br>
                Cumple con Casco y Chaleco
                </div>
                """, unsafe_allow_html=True)
            else:
                faltantes = requeridos - presentes
                st.markdown(f"""
                <div style='background-color:#fee2e2; padding:15px; border-radius:10px;'>
                🔴 <b>ACCESO DENEGADO</b><br>
                Faltan: {', '.join(faltantes)}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📊 Resultados de detección")

            if datos_analitica:
                df_analitica = pd.DataFrame(datos_analitica)
                df_analitica = df_analitica.sort_values(by="Confianza", ascending=False).reset_index(drop=True)
                st.dataframe(df_analitica, use_container_width=True)
            else:
                st.warning("⚠️ No se detectó ningún equipo de protección.")

        st.markdown("---")

# -----------------------
# FOOTER
# -----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px; color:#6b7280;'>
Desarrollado por <b>Mateo Sandoval</b> • Ingeniería de Sistemas • 2026
</p>
""", unsafe_allow_html=True)
