"""ZONA LIBRE: interfaz de GlowUp Radar para la gerencia de Seoru.
Usa la zona fija (core/) para cargar el modelo, validar, predecir, decidir y explicar.
Para verla en su computador:   streamlit run app.py"""
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import cargar_config, cargar_metadata, cargar_modelo, cargar_resultados, etiqueta, evaluar, hay_modelo, validar

config = cargar_config()
st.set_page_config(page_title=config["titulo"], page_icon="💧", layout="wide")

COLOR = {"Firmar": "#2E7D32", "Firmar con garantías": "#E09B00", "No firmar": "#C62828"}
QUE_HACER = {
    "Firmar": "Riesgo bajo de abandono. Se puede negociar un contrato estándar.",
    "Firmar con garantías": ("Riesgo medio. Negociar solo con garantías: compras mínimas, cesión de las notificaciones "
                             "sanitarias a quien Seoru designe si el contrato termina o pago anticipado del primer pedido."),
    "No firmar": "Riesgo alto de que reduzca sus importaciones a menos de la mitad el próximo año. No priorizar.",
}
# Variables de LegisComex: se nombran en las razones pero nunca se muestra su valor exacto
SIN_VALOR = {"log_cif", "n_decl_t", "n_proveedores_t", "n_paises_t", "n_meses_activos_t", "share_china_t",
             "share_ue_t", "share_usa_t", "momentum_h2_t", "usd_por_kg_t", "intensidad_sanitaria",
             "log_valor_por_mes", "decl_por_proveedor"}


def pesos(valor):
    signo = "−" if valor < 0 else ""
    return f"{signo}US${abs(valor):,.0f}".replace(",", ".")


def md(texto):
    """Escapa el signo $ para que Streamlit no lo lea como fórmula."""
    return texto.replace("$", "\\$")


def chip(decision):
    return (f"<span style='background:{COLOR.get(decision, '#555')};color:white;padding:3px 12px;"
            f"border-radius:999px;font-weight:600'>{decision}</span>")


# ---------------------------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------------------------
col_logo, col_titulo = st.columns([2, 7], vertical_alignment="center")
with col_logo:
    try:
        st.image(config["logo"], width=220)
    except Exception:
        pass
with col_titulo:
    st.title(config["titulo"])
    st.caption(f"{config['cliente']} · {config['descripcion']}")

with st.container(border=True):
    st.markdown("#### ¿Qué es GlowUp Radar?")
    st.markdown(
        "Seoru necesita un importador colombiano para vender sus productos en Colombia. Si escoge uno que al año "
        "siguiente casi deja de importar (**abandona**), pierde el canal de venta y su inversión. GlowUp Radar analiza "
        "a los **540 importadores de cosméticos de 2025** y estima, para cada uno, **qué tan probable es que abandone "
        "en 2026**.")
    i1, i2, i3, i4 = st.columns(4)
    i1.markdown("**1. Mide el riesgo**  \nMira cuántos meses al año importa, cuánto importa, sus registros en el "
                "INVIMA y la antigüedad de la empresa.")
    i2.markdown("**2. Recomienda**  \n🟢 **Firmar**: riesgo bajo  \n🟡 **Con garantías**: riesgo medio  \n"
                "🔴 **No firmar**: riesgo alto")
    i3.markdown("**3. Escoge 20 candidatos**  \nEntre los confiables, prioriza a quienes venden productos parecidos "
                "a los de Seoru (sérums, limpiadores, protector solar facial).")
    i4.markdown("**4. Explica por qué**  \nPara cada importador muestra las razones que más suben o bajan su "
                "riesgo, para que Seoru entienda la recomendación.")

if not hay_modelo():
    st.info("Todavía no hay modelo en la carpeta model/. Exporten el modelo con exportar (README, PASO 3).")
    st.divider()
    st.caption(config["pie"])
    st.stop()


@st.cache_resource
def recursos():
    return cargar_metadata(), cargar_modelo(), cargar_resultados()


meta, modelo, resultados = recursos()
m = meta["metricas"]
u1, u2 = meta["umbrales"]


@st.cache_data
def tabla_radar():
    t = resultados.copy()
    t["nit"] = t["nit"].astype(str)
    t["prob_pct"] = (100 * t["probabilidad"]).round(1)
    t["busqueda"] = t["razon_social"].astype(str) + " · NIT " + t["nit"]
    orden_tam = ["US$0 a US$100.000", "US$100.000 a US$500.000", "US$500.000 a US$2.000.000",
                 "US$2.000.000 a US$10.000.000", "más de US$10.000.000"]
    t["rango_cif_t"] = pd.Categorical(t["rango_cif_t"], categories=orden_tam, ordered=True)
    return t.sort_values("probabilidad", ascending=True).reset_index(drop=True)


radar = tabla_radar()


def razones_csv(texto):
    """'Meses con importaciones baja 2.8 pts' -> (variable, sube/baja, puntos)."""
    salida = []
    for parte in str(texto).split(" · "):
        r = re.match(r"(.+?) (sube|baja) ([\d.]+) pts$", parte.strip())
        if r:
            salida.append((r.group(1), r.group(2), float(r.group(3))))
    return salida


def mostrar_razones(lista):
    if not lista:
        st.caption("Con estos datos ninguna variable mueve la probabilidad frente a un importador típico.")
        return
    for nombre, sentido, puntos in lista:
        icono = "🔺" if sentido == "sube" else "🔻"
        efecto = "aumenta" if sentido == "sube" else "reduce"
        puntos_txt = f"{puntos:.1f}".replace(".", ",")
        st.markdown(f"{icono} **{nombre}** {efecto} el riesgo de abandono en **{puntos_txt} puntos**")
    st.caption("Cada razón compara a este importador con uno típico (valores medianos). Dice de qué depende la "
               "predicción del modelo, no qué pasaría si el importador cambiara esa variable.")


def tarjeta_resultado(prob, decision, ranking=None):
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad de abandono", f"{prob:.0%}", help="Probabilidad de que el importador reduzca sus "
              "importaciones de cosméticos a menos de la mitad el año siguiente.")
    with c2:
        st.markdown("**Recomendación**")
        st.markdown(chip(decision), unsafe_allow_html=True)
    st.info(QUE_HACER.get(decision, ""))
    if ranking is not None and not pd.isna(ranking):
        st.success(f"⭐ Está en el puesto **{int(ranking)}** de los 20 candidatos para diligencia debida.")


# ---------------------------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------------------------
tabs = st.tabs(["🔎 Consultar un importador", "📡 Radar 2026 y 20 candidatos", "💰 Cuánto vale el radar",
                "⚙️ Cómo decide", "✏️ Evaluar uno nuevo"])

# 1. Consultar ----------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Consultar un importador de 2025")
    texto = st.text_input("Escriban el nombre o el NIT (o parte de él)", placeholder="Ej.: Galderma, 830012269")
    opciones = radar
    if texto.strip():
        patron = texto.strip().upper()
        opciones = radar[radar["busqueda"].str.upper().str.contains(re.escape(patron))]
    if opciones.empty:
        st.warning("No encontramos ese importador en el radar 2026. Revisen la escritura o usen la pestaña "
                   "«Evaluar uno nuevo».")
    else:
        elegido = st.selectbox(f"{len(opciones)} importador(es) encontrados", opciones["busqueda"].tolist())
        fila = radar[radar["busqueda"] == elegido].iloc[0]
        st.markdown(f"### {fila['razon_social']}")
        st.caption(f"NIT {fila['nit']}")
        izq, der = st.columns([3, 2])
        with izq:
            tarjeta_resultado(float(fila["probabilidad"]), fila["decision"], fila.get("ranking_diligencia"))
            st.markdown("**Razones principales de la recomendación**")
            mostrar_razones(razones_csv(fila["razones"]))
        with der:
            st.markdown("**Perfil del importador (2025)**")
            perfil = pd.DataFrame({
                "Dato": ["Valor importado al año", "Meses con importaciones", "Producto principal notificado",
                         "Afinidad con la línea Seoru", "Antigüedad de la empresa", "Actividad económica",
                         "Tiene notificaciones INVIMA"],
                "Valor": [str(fila["rango_cif_t"]), fila["rango_meses_activos"], fila["producto_principal"],
                          f"{fila['afinidad_seoru_pct']:.0f} % de sus productos", f"{fila['antiguedad_anios']:.0f} años",
                          fila["grupo_actividad"], fila["tiene_notificaciones_invima"]],
            })
            st.dataframe(perfil, hide_index=True, width="stretch")
            st.caption("Los valores de importación se muestran en rangos por las condiciones de uso de los datos.")

# 2. Radar --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Radar 2026: importadores de 2025 y su recomendación")
    conteo = radar["decision"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Importadores evaluados", len(radar))
    k2.metric("Firmar", int(conteo.get("Firmar", 0)))
    k3.metric("Firmar con garantías", int(conteo.get("Firmar con garantías", 0)))
    k4.metric("No firmar", int(conteo.get("No firmar", 0)))

    solo_20 = st.toggle("Ver solo los 20 candidatos para diligencia debida", value=False)
    st.caption("**Cómo se eligen los 20:** entre los importadores con recomendación «Firmar» o «Firmar con "
               "garantías», se ordenan por afinidad con la línea Seoru (proporción de productos notificados con "
               "sérum, niacinamida, ácido hialurónico, limpiador o protector solar facial) y, en empate, por menor probabilidad de abandono.")
    f1, f2, f3 = st.columns(3)
    decisiones_sel = f1.multiselect("Recomendación", meta["decisiones"], default=meta["decisiones"])
    tamanos = [c for c in radar["rango_cif_t"].cat.categories if (radar["rango_cif_t"] == c).any()]
    tamanos_sel = f2.multiselect("Tamaño (valor importado al año)", tamanos, default=tamanos)
    productos = sorted(radar["producto_principal"].dropna().unique())
    productos_sel = f3.multiselect("Producto principal notificado", productos, default=productos)
    solo_afinidad = st.checkbox("Solo importadores con productos parecidos a los de Seoru (afinidad > 0)")

    vista = radar[radar["decision"].isin(decisiones_sel) & radar["rango_cif_t"].isin(tamanos_sel)
                  & radar["producto_principal"].isin(productos_sel)]
    if solo_afinidad:
        vista = vista[vista["afinidad_seoru_pct"] > 0]
    if solo_20:
        vista = vista[vista["ranking_diligencia"].notna()].sort_values("ranking_diligencia")
    st.write(f"**{len(vista)}** importadores")
    columnas = {"ranking_diligencia": "Puesto (20)", "razon_social": "Importador", "nit": "NIT",
                "prob_pct": "Prob. abandono (%)", "decision": "Recomendación", "afinidad_seoru_pct": "Afinidad Seoru (%)",
                "producto_principal": "Producto principal", "rango_cif_t": "Valor importado",
                "rango_meses_activos": "Meses activos", "razones": "Razones principales"}
    mostrar_tabla = vista[list(columnas)].copy()
    mostrar_tabla["ranking_diligencia"] = [f"⭐ {int(v)}" if pd.notna(v) else "" for v in mostrar_tabla["ranking_diligencia"]]
    marcas = {"Firmar": "🟢 Firmar", "Firmar con garantías": "🟡 Firmar con garantías", "No firmar": "🔴 No firmar"}
    mostrar_tabla["decision"] = mostrar_tabla["decision"].map(marcas)
    st.dataframe(
        mostrar_tabla.rename(columns=columnas), hide_index=True, width="stretch", height=520,
        column_config={
            "Afinidad Seoru (%)": st.column_config.NumberColumn(format="%.0f %%"),
            "Prob. abandono (%)": st.column_config.NumberColumn(format="%.1f %%"),
        })
    st.download_button("Descargar esta lista (CSV)", vista[list(columnas)].rename(columns=columnas)
                       .to_csv(index=False).encode("utf-8-sig"), "radar_2026_seoru.csv", "text/csv")

# 3. Valor --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("¿Cuánto vale el radar frente a la regla simple?")
    ahorro = m["costo_regla"] - m["costo_modelo"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Costo de los errores con la regla simple", pesos(m["costo_regla"]))
    c2.metric("Costo de los errores con el radar", pesos(m["costo_modelo"]))
    c3.metric("Ahorro del radar frente a la regla", pesos(ahorro))
    st.caption(md(f"Costos {m['unidad_costo']}. Regla simple: {m['regla_simple']}."))

    datos_g = pd.DataFrame({"Estrategia": ["Regla simple (meses activos)", "Radar (Gradient Boosting)"],
                            "Costo": [m["costo_regla"], m["costo_modelo"]]})
    st.altair_chart(alt.Chart(datos_g).mark_bar(size=38).encode(
        x=alt.X("Costo:Q", title="US$ por cada 100 importadores"), y=alt.Y("Estrategia:N", title=None),
        color=alt.Color("Estrategia:N", legend=None, scale=alt.Scale(domain=datos_g["Estrategia"].tolist(),
                                                                       range=["#9E9E9E", "#2A2C7A"])),
        tooltip=[alt.Tooltip("Costo:Q", format=",.0f")]).properties(height=160), width="stretch")

    a, b = m["intervalo_diferencia"]
    if ahorro < 0:
        st.warning(md(
            f"**Lectura honesta:** en la prueba de 2024 el radar costó {pesos(-ahorro)} más que la regla simple por "
            f"cada 100 importadores. El intervalo de 95 % del ahorro va de {pesos(a)} a {pesos(b)}: incluye el cero, "
            f"así que **no podemos afirmar que uno sea mejor que el otro en costo**. Solo en el "
            f"{m['pct_muestras_modelo_gana']:.0%} de las muestras bootstrap el radar sale más barato."))
    else:
        st.success(md(f"El radar ahorra {pesos(ahorro)} por cada 100 importadores (intervalo de 95 %: {pesos(a)} a {pesos(b)})."))

    st.markdown("**Entonces, ¿para qué sirve el radar si no es más barato?** Aporta cuatro cosas que la regla no da:")
    st.markdown(
        f"- **Distingue el riesgo medio:** la regla solo dice firmar o no firmar; el radar identifica a quién pedirle "
        f"garantías ({int((radar['decision'] == 'Firmar con garantías').sum())} importadores en 2025).\n"
        f"- **Ordena a los candidatos:** combina el riesgo con la afinidad con Seoru para escoger los 20 de la "
        f"diligencia debida; la regla deja a muchos empatados.\n"
        f"- **Explica cada recomendación** con razones, como pidió la Jefa de Asuntos Regulatorios.\n"
        f"- **Ordena un poco mejor por riesgo:** AUC {m['auc_modelo']:.3f} del radar frente a {m['auc_regla']:.3f} de la regla."
        .replace("0.", "0,"))
    tabla_m = pd.DataFrame({
        "Indicador (prueba 2024)": ["Costo por 100 importadores", "Sensibilidad (abandonos detectados)",
                                    "Abandonos con los que se firmaría sin garantías", "AUC"],
        "Regla simple": [pesos(m["costo_regla"]), f"{m['sensibilidad_regla']:.1%}", str(m["abandonos_que_firma_regla"]),
                         f"{m['auc_regla']:.3f}".replace(".", ",")],
        "Radar": [pesos(m["costo_modelo"]), f"{m['sensibilidad_modelo']:.1%}", str(m["abandonos_que_firma_modelo"]),
                  f"{m['auc_modelo']:.3f}".replace(".", ",")],
    })
    st.dataframe(tabla_m, hide_index=True, width="stretch")
    st.caption(md(f"Medido con {m['importadores_prueba']} importadores de 2024 que el modelo no vio al entrenarse. "
               "Costos del caso: firmar con quien abandona US$250.000; no firmar con quien se habría quedado US$50.000; "
               "garantías con quien se queda US$20.000 y con quien abandona US$120.000."))

# 4. Cómo decide --------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Cómo convierte el radar una probabilidad en una recomendación")
    regla = pd.DataFrame({
        "Probabilidad de abandono": [f"menos de {u1:.0%}", f"{u1:.0%} a {u2:.0%}", f"{u2:.0%} o más"],
        "Recomendación": meta["decisiones"],
    })
    st.dataframe(regla, hide_index=True, width="stretch")
    st.markdown(
        "- **Qué predice:** la probabilidad de que el importador reduzca sus importaciones de cosméticos (capítulo 33) "
        "a menos de la mitad el año siguiente.\n"
        "- **Modelo:** Gradient Boosting (200 árboles de profundidad 2), elegido por tener el menor costo en "
        "validación cruzada con 2023. Para el radar 2026 se reentrenó con 2023 y 2024 "
        f"({meta['n_entrenamiento']} importadores).\n"
        "- **Umbrales:** se eligieron minimizando el costo de los errores en validación cruzada (2023).\n"
        "- **Variables en las que más se apoya:** meses con importaciones y peso del segundo semestre "
        "(importancia por permutación y SHAP). Son asociaciones, no causas.\n"
        "- **Población:** importadores con NIT que tienen notificaciones INVIMA o actividad de comercio (CIIU 46 o 47).")
    st.markdown("**Límites:** el abandono se mide con todas las importaciones del capítulo 33, no con una marca en "
                "particular; los intervalos son anchos (≈100 abandonos en la prueba); las condiciones del mercado de "
                "2026 pueden ser distintas a las de 2023-2024.")
    st.caption(f"Modelo exportado el {meta['fecha_exportacion']}.")

# 5. Evaluar uno nuevo --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Evaluar un importador que no está en el radar")
    st.caption("Escriban los datos del importador para un año. Los campos que no conozcan quedan con el valor "
               "típico de los importadores con que se entrenó el modelo.")
    ref = {c["nombre"]: c.get("referencia") for c in meta["columnas"]}
    cats = {c["nombre"]: c.get("categorias") for c in meta["columnas"] if c["tipo"] == "categórica"}
    with st.form("nuevo"):
        a1, a2, a3 = st.columns(3)
        cif = a1.number_input("Valor importado en el año (US$ CIF)", min_value=1000.0, value=float(round(10 ** ref["log_cif"], -3)),
                              step=10000.0)
        meses = a2.number_input("Meses del año con importaciones", 1, 12, int(ref["n_meses_activos_t"]))
        decl = a3.number_input("Número de declaraciones de importación", 1, 20000, int(ref["n_decl_t"]))
        b1, b2, b3 = st.columns(3)
        prov = b1.number_input("Número de proveedores extranjeros", 1, 500, int(ref["n_proveedores_t"]))
        paises = b2.number_input("Países de origen", 1, 60, int(ref["n_paises_t"]))
        edad = b3.number_input("Antigüedad de la empresa (años)", 0.0, 120.0, float(round(ref["edad_anios_t"], 1)))
        c1_, c2_, c3_ = st.columns(3)
        reg = c1_.number_input("Notificaciones INVIMA nuevas en el año", 0, 2000, int(ref["reg_t"]))
        reg_ant = c2_.number_input("Notificaciones INVIMA nuevas el año anterior", 0, 2000, int(ref["reg_t"]))
        afin = c3_.slider("Afinidad con Seoru (% de productos de la línea Seoru)", 0, 100, int(100 * ref["afinidad_seoru"]))
        d1, d2 = st.columns(2)
        forma = d1.selectbox("Forma jurídica", cats["forma"])
        actividad = d2.selectbox("Actividad económica", cats["grupo_actividad"])
        with st.expander("Más variables (opcional)"):
            e1, e2, e3 = st.columns(3)
            china = e1.slider("% del valor desde China", 0, 100, int(100 * ref["share_china_t"]))
            ue = e2.slider("% del valor desde la Unión Europea", 0, 100, int(100 * ref["share_ue_t"]))
            usa = e3.slider("% del valor desde EE. UU.", 0, 100, int(100 * ref["share_usa_t"]))
            g1, g2, g3 = st.columns(3)
            h2 = g1.slider("% del valor importado en el 2.º semestre", 0, 100, int(100 * ref["momentum_h2_t"]))
            kilo = g2.number_input("Dólares por kilo", 0.1, 20000.0, float(round(ref["usd_por_kg_t"], 1)))
            fabs = g3.number_input("Fabricantes distintos en sus notificaciones", 0, 500, int(ref["n_fabricantes_t"]))
            h1_, h2_, h3_ = st.columns(3)
            fab_ext = h1_.slider("% de notificaciones con fabricante extranjero", 0, 100, int(100 * ref["share_fab_extranjero_t"]))
            ncat = h2_.number_input("Categorías de producto distintas (de 10)", 0, 10, int(ref["n_categorias_t"]))
            bogota = h3_.selectbox("Matrícula en la Cámara de Comercio de Bogotá", ["Sí", "No"])
            st.markdown("Proporción de sus notificaciones en cada categoría (%)")
            i1, i2, i3, i4, i5 = st.columns(5)
            skin = i1.number_input("Cuidado de la piel", 0, 100, 0)
            maqui = i2.number_input("Maquillaje", 0, 100, 0)
            capi = i3.number_input("Capilar", 0, 100, 0)
            corp = i4.number_input("Corporal", 0, 100, 0)
            solar = i5.number_input("Solar", 0, 100, 0)
        enviado = st.form_submit_button("Evaluar importador", type="primary")
    if enviado:
        fila_nueva = {
            "log_cif": np.log10(cif), "n_decl_t": decl, "n_proveedores_t": prov, "n_paises_t": paises,
            "n_meses_activos_t": meses, "share_china_t": china / 100, "share_ue_t": ue / 100, "share_usa_t": usa / 100,
            "momentum_h2_t": h2 / 100, "usd_por_kg_t": kilo, "reg_t": reg, "reg_crec": (reg + 1) / (reg_ant + 1),
            "share_fab_extranjero_t": fab_ext / 100, "n_fabricantes_t": fabs, "cat_skincare": skin / 100,
            "cat_maquillaje": maqui / 100, "cat_capilar": capi / 100, "cat_corporal": corp / 100,
            "cat_solar": solar / 100, "n_categorias_t": ncat, "edad_anios_t": edad,
            "camara_bogota": 1 if bogota == "Sí" else 0, "intensidad_sanitaria": reg / (cif / 1e6),
            "log_valor_por_mes": np.log10(cif / meses), "decl_por_proveedor": decl / prov,
            "afinidad_seoru": afin / 100, "forma": forma, "grupo_actividad": actividad,
        }
        datos, errores, avisos = validar(pd.DataFrame([fila_nueva]), meta)
        for e in errores:
            st.error(e)
        for aviso in avisos:
            st.warning(aviso)
        if not errores:
            r = evaluar(modelo, meta, datos)[0]
            tarjeta_resultado(r["probabilidad"], r["decision"])
            st.markdown("**Razones principales**")
            mostrar_razones([(etiqueta(meta, x["variable"]), x["sentido"], abs(x["efecto"]) * 100) for x in r["razones"]])


# Pie de página: debe quedar (atribuciones y advertencias de su caso, en mi_app.py)
st.divider()
st.caption(config["pie"])
