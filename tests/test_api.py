"""
Test end-to-end:
- crea dataset mini en carpeta temporal
- entrena y guarda modelo/metrics en tmp
- setea env vars MODEL_PATH/METRICS_PATH para que la API los use
- recarga el módulo app.main (para que ejecute startup con esos paths)
- llama a /predict y valida respuesta
"""

from __future__ import annotations

import tempfile
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient

from src.train import train_and_save


def test_predict_after_training(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # paths en temp
        model_path = tmp / "ticket_model.joblib"
        metrics_path = tmp / "metrics.json"
        data_path = tmp / "tickets.csv"

        # dataset mínimo (6 ejemplos, 6 clases)
        data_path.write_text(
            "text,label\n"
            # account_access (5)
            "No puedo iniciar sesión,account_access\n"
            "He olvidado mi password,account_access\n"
            "Mi cuenta está bloqueada,account_access\n"
            "No me llega el email de verificación,account_access\n"
            "Quiero activar 2FA,account_access\n"
            # billing (5)
            "Me han cobrado dos veces,billing\n"
            "No entiendo este cargo en mi factura,billing\n"
            "Quiero cambiar mi método de pago,billing\n"
            "La factura tiene un importe incorrecto,billing\n"
            "Necesito un duplicado de la factura,billing\n"
            # cancellation (5)
            "Quiero darme de baja,cancellation\n"
            "Deseo cancelar mi suscripción,cancellation\n"
            "No quiero renovar el plan,cancellation\n"
            "Cómo cancelo mi contrato,cancellation\n"
            "Quiero cerrar mi cuenta,cancellation\n"
            # shipping_delivery (5)
            "Mi pedido no ha llegado,shipping_delivery\n"
            "El tracking no se actualiza,shipping_delivery\n"
            "El envío se retrasó,shipping_delivery\n"
            "Recibí el paquete dañado,shipping_delivery\n"
            "La dirección de entrega está mal,shipping_delivery\n"
            # technical_support (5)
            "La app se cierra al iniciar,technical_support\n"
            "No puedo conectarme al servicio,technical_support\n"
            "Error 500 al guardar cambios,technical_support\n"
            "La página va muy lenta desde ayer,technical_support\n"
            "No recibo notificaciones,technical_support\n"
            # general_inquiry (5)
            "Qué planes tenéis,general_inquiry\n"
            "Necesito información sobre precios,general_inquiry\n"
            "Cómo funciona el servicio,general_inquiry\n"
            "Tenéis descuentos para estudiantes,general_inquiry\n"
            "Horario de atención al cliente,general_inquiry\n",
            encoding="utf-8",
        )

        # entrena y guarda modelo/metrics
        train_and_save(data_path, model_path, metrics_path)

        # forzamos a la API a leer estos paths
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("METRICS_PATH", str(metrics_path))

        # recargamos módulo para que tome env vars y cargue modelo
        import src.inference as inf_mod

        reload(inf_mod)

        import app.main as main_mod

        reload(main_mod)

        with TestClient(main_mod.app) as client:
            r = client.post("/predict", json={"text": "No puedo iniciar sesión con mi contraseña"})
            assert r.status_code == 200

        body = r.json()
        assert "label" in body
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0
