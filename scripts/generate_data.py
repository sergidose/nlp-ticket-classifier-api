"""
Genera un dataset sintético (reproducible) de tickets de soporte.
Ventaja: no dependes de Kaggle ni de descargas externas.
Salida: data/raw/tickets.csv (columnas: text, label)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Raíz del repo (…/nlp-ticket-classifier-api)
ROOT = Path(__file__).resolve().parents[1]

# Donde guardaremos el dataset generado
OUT_PATH = ROOT / "data" / "raw" / "tickets.csv"

# Frases "plantilla" por categoría (labels)
CATEGORIES = {
    "billing": [
        "No entiendo este cargo en mi factura",
        "Me han cobrado dos veces este mes",
        "Quiero cambiar mi método de pago",
        "La factura tiene un importe incorrecto",
        "Necesito un duplicado de la factura",
    ],
    "technical_support": [
        "La app se cierra al iniciar",
        "No puedo conectarme al servicio",
        "Error 500 al guardar cambios",
        "La página va muy lenta desde ayer",
        "No recibo notificaciones",
    ],
    "account_access": [
        "No puedo iniciar sesión con mi contraseña",
        "He olvidado mi password",
        "No me llega el email de verificación",
        "Mi cuenta está bloqueada",
        "Quiero activar 2FA",
    ],
    "cancellation": [
        "Quiero darme de baja del servicio",
        "Deseo cancelar mi suscripción",
        "No quiero renovar el plan",
        "Cómo cancelo mi contrato",
        "Quiero cerrar mi cuenta",
    ],
    "shipping_delivery": [
        "Mi pedido no ha llegado",
        "El tracking no se actualiza",
        "Recibí el paquete dañado",
        "El envío se retrasó",
        "La dirección de entrega está mal",
    ],
    "general_inquiry": [
        "Qué planes tenéis disponibles",
        "Necesito información sobre precios",
        "Cómo funciona el servicio",
        "Tenéis descuentos para estudiantes",
        "Horario de atención al cliente",
    ],
}


def generate(n_per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Crea un DataFrame con (text, label). Se “baraja” y se añade una pequeña variación
    para que no sean frases 100% idénticas.
    """
    # serie aleatoria para introducir un “id” y variar un poco el texto
    rng = (
        pd.Series(range(n_per_class * len(CATEGORIES)))
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    rows = []
    idx = 0

    for label, templates in CATEGORIES.items():
        for _ in range(n_per_class):
            base = templates[idx % len(templates)]
            text = f"{base}. Ticket #{int(rng.iloc[idx])}"
            rows.append({"text": text, "label": label})
            idx += 1

    # baraja final para mezclar categorías
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    # crea carpetas si no existen
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
