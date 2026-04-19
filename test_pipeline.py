"""
test_pipeline.py
================
Unit tests for the Lifeline ECG Vision API — non-AI pipeline only.

Rules
-----
- Zero real API calls. All external clients (requests, Groq, Azure) are mocked
  via unittest.mock.patch.
- importlib.reload() is used after env-var patches so modules re-read os.environ.
- Tests cover: exception handling, error propagation, routing logic, auth checks,
  and YOLO early-exit guards.

Run with:
    pytest test_pipeline.py -v
"""
from __future__ import annotations

import io
import sys
import json
import types
import importlib
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock, AsyncMock


# ============================================================
# Helper utilities
# ============================================================
def _make_png_bytes(width: int = 100, height: int = 80) -> bytes:
    """Return raw PNG bytes for a solid-white RGB image."""
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_pil_image(width: int = 100, height: int = 80) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


# ============================================================
# 1. YOLO_Detector — crop_ecg_from_bytes
# ============================================================
class TestYOLODetector:
    """Tests for crop_ecg_from_bytes, YOLO weights never loaded."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Import once; mock the module-level yolo_model in place."""
        if "YOLO_Detector" not in sys.modules:
            # First import — YOLO will try to load weights; stub the YOLO class
            with patch("ultralytics.YOLO"):
                import YOLO_Detector as _mod
        else:
            import YOLO_Detector as _mod

        self._mod = _mod
        # Replace the module-level model with a fresh MagicMock each test
        self._yolo = MagicMock()
        self._mod.yolo_model = self._yolo
        yield

    # ---- YOLO detects a box → returns PIL Image -----------------------------
    def test_successful_crop_returns_pil_image(self):
        # YOLO_Detector adds 80px padding; the box coordinates returned by YOLO
        # on the padded canvas must exceed padding to yield a valid crop area.
        box = MagicMock()
        box.xyxy = [MagicMock()]
        # padded canvas coords: x1=90, y1=90, x2=190, y2=160  (well inside 260x240 padded image)
        box.xyxy[0].cpu.return_value.numpy.return_value = [90, 90, 190, 160]
        result_obj = MagicMock()
        result_obj.boxes = [box]
        self._yolo.predict.return_value = [result_obj]

        result = self._mod.crop_ecg_from_bytes(_make_png_bytes())
        assert isinstance(result, Image.Image)

    # ---- Both padded & fallback predict return no boxes → error dict --------
    def test_no_detection_returns_error_dict(self):
        empty = MagicMock()
        empty.boxes = []
        self._yolo.predict.return_value = [empty]

        result = self._mod.crop_ecg_from_bytes(_make_png_bytes())
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "message" in result

    # ---- RGBA PNG is flattened to RGB without crash --------------------------
    def test_rgba_image_handled(self):
        box = MagicMock()
        box.xyxy = [MagicMock()]
        # Same corrected padding-aware coordinates
        box.xyxy[0].cpu.return_value.numpy.return_value = [90, 90, 190, 160]
        result_obj = MagicMock()
        result_obj.boxes = [box]
        self._yolo.predict.return_value = [result_obj]

        rgba = Image.new("RGBA", (100, 80), (255, 255, 255, 128))
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")

        result = self._mod.crop_ecg_from_bytes(buf.getvalue())
        assert isinstance(result, Image.Image)

    # ---- Corrupt bytes → exception raised (PIL can't open) ------------------
    def test_corrupt_bytes_raises(self):
        with pytest.raises(Exception):
            self._mod.crop_ecg_from_bytes(b"not an image at all")


# ============================================================
# 2. LLAVA_FineTuned — analyze_ecg_with_pulse
# ============================================================
class TestLLAVAFineTuned:
    """HTTP calls fully mocked; tests error handling contracts."""

    @pytest.fixture(autouse=True)
    def _reload(self):
        if "LLAVA_FineTuned" in sys.modules:
            del sys.modules["LLAVA_FineTuned"]
        import LLAVA_FineTuned
        self._mod = LLAVA_FineTuned
        yield

    def test_successful_response_parsed(self):
        payload = {"overall_interpretation": "Normal", "findings": [], "summary_report": "OK"}
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": json.dumps(payload)})
            )
            result = self._mod.analyze_ecg_with_pulse(_make_pil_image())
        assert result["overall_interpretation"] == "Normal"

    def test_empty_analysis_returns_safe_fallback(self):
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": ""})
            )
            result = self._mod.analyze_ecg_with_pulse(_make_pil_image())
        assert result.get("overall_interpretation") == "Error"
        assert "summary_report" in result

    def test_connection_error_returns_error_dict(self):
        import requests
        with patch.object(self._mod.requests, "post",
                          side_effect=requests.exceptions.ConnectionError("refused")):
            result = self._mod.analyze_ecg_with_pulse(_make_pil_image())
        assert result.get("overall_interpretation") == "Error"

    def test_http_500_returns_error_dict(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with patch.object(self._mod.requests, "post", return_value=mock_resp):
            result = self._mod.analyze_ecg_with_pulse(_make_pil_image())
        assert result.get("overall_interpretation") == "Error"

    def test_malformed_json_in_analysis_returns_error_dict(self):
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": "{broken{{json"})
            )
            result = self._mod.analyze_ecg_with_pulse(_make_pil_image())
        assert result.get("overall_interpretation") == "Error"


# ============================================================
# 3. LLAVA_Prompt — query_llava_2
# ============================================================
class TestLLAVAPrompt:
    @pytest.fixture(autouse=True)
    def _reload(self):
        if "LLAVA_Prompt" in sys.modules:
            del sys.modules["LLAVA_Prompt"]
        import LLAVA_Prompt
        self._mod = LLAVA_Prompt
        yield

    def test_text_only_returns_dict(self):
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": json.dumps({"raw_text_response": "Normal"})})
            )
            result = self._mod.query_llava_2("Is this ECG normal?")
        assert isinstance(result, dict)

    def test_image_and_text_returns_dict(self):
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": json.dumps({"finding": "AF"})})
            )
            result = self._mod.query_llava_2("Analyze ECG", _make_pil_image())
        assert isinstance(result, dict)

    def test_plain_text_response_uses_fallback_key(self):
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"analysis": "This is plain text, not JSON"})
            )
            result = self._mod.query_llava_2("Analyze ECG")
        assert "raw_text_response" in result

    def test_timeout_error_returns_error_dict(self):
        import requests
        with patch.object(self._mod.requests, "post",
                          side_effect=requests.exceptions.Timeout("timeout")):
            result = self._mod.query_llava_2("Analyze ECG")
        assert "error" in result

    def test_missing_analysis_key_does_not_crash(self):
        """Bridge returns {} (missing 'analysis') → raw_text_response fallback."""
        with patch.object(self._mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={})
            )
            result = self._mod.query_llava_2("Analyze ECG")
        assert isinstance(result, dict)


# ============================================================
# 4. GPTNano_Context — generate_clinical_summary
# ============================================================
class TestGPTNanoContext:
    @pytest.fixture(autouse=True)
    def _reload(self):
        if "GPTNano_Context" in sys.modules:
            del sys.modules["GPTNano_Context"]
        import GPTNano_Context
        self._mod = GPTNano_Context
        yield

    def test_missing_token_returns_error_string(self):
        """Patch os.environ inside the module so the cached real key is hidden."""
        import GPTNano_Context as _mod
        with patch.object(_mod.os, "environ", {}):
            result = _mod.generate_clinical_summary({}, {})
        assert "Error" in result and "GITHUB_TOKEN" in result

    def test_azure_exception_returns_error_string(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake"}):
            importlib.reload(self._mod)
            mock_client = MagicMock()
            mock_client.complete.side_effect = Exception("Azure down")
            with patch.object(self._mod, "ChatCompletionsClient", return_value=mock_client):
                result = self._mod.generate_clinical_summary({"finding": "AF"}, {"result": "Abnormal"})
        assert "Could not generate summary" in result

    def test_successful_call_returns_non_empty_string(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake"}):
            importlib.reload(self._mod)
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "1. Disease: AF 2. Action: Call doctor"
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_response
            with patch.object(self._mod, "ChatCompletionsClient", return_value=mock_client):
                result = self._mod.generate_clinical_summary({"finding": "AF"}, {"result": "Abnormal"})
        assert isinstance(result, str) and len(result) > 0


# ============================================================
# 5. GPTNano_Prompt — generate_clinical_summary_2
# ============================================================
class TestGPTNanoPrompt:
    @pytest.fixture(autouse=True)
    def _reload(self):
        if "GPTNano_Prompt" in sys.modules:
            del sys.modules["GPTNano_Prompt"]
        import GPTNano_Prompt
        self._mod = GPTNano_Prompt
        yield

    def test_missing_token_returns_error_string(self):
        """Patch os.environ inside the module so the cached real key is hidden."""
        import GPTNano_Prompt as _mod
        with patch.object(_mod.os, "environ", {}):
            result = _mod.generate_clinical_summary_2("Analyze this ECG", None)
        assert "Error" in result

    def test_azure_exception_returns_error_string(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake"}):
            importlib.reload(self._mod)
            mock_client = MagicMock()
            mock_client.complete.side_effect = RuntimeError("Rate limited")
            with patch.object(self._mod, "ChatCompletionsClient", return_value=mock_client):
                result = self._mod.generate_clinical_summary_2("prompt", "context")
        assert "Could not generate summary" in result

    def test_none_context_does_not_crash(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake"}):
            importlib.reload(self._mod)
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Summary"
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_response
            with patch.object(self._mod, "ChatCompletionsClient", return_value=mock_client):
                result = self._mod.generate_clinical_summary_2("Analyze ECG", None)
        assert isinstance(result, str)

    def test_dict_llava_input_stringified_without_crash(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake"}):
            importlib.reload(self._mod)
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Structured report"
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_response
            with patch.object(self._mod, "ChatCompletionsClient", return_value=mock_client):
                result = self._mod.generate_clinical_summary_2({"key": "value"}, "context")
        assert isinstance(result, str)


# ============================================================
# 6. Groq_Summary — generate_master_consensus
# ============================================================
class TestGroqSummary:
    @pytest.fixture(autouse=True)
    def _reload(self):
        if "Groq_Summary" in sys.modules:
            del sys.modules["Groq_Summary"]
        import Groq_Summary
        self._mod = Groq_Summary
        yield

    def test_groq_exception_returns_error_string(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("Groq unavailable")
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus({"finding": "AF"}, {"r": "Abnormal"}, "GPT summary")
        assert "Could not generate final consensus" in result

    def test_successful_call_returns_string(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Final clinical report"
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_completion
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus({}, {}, "GPT summary")
        assert "Final clinical report" in result

    def test_medgemma_as_string_does_not_crash(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Report"
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_completion
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus({}, "plain string medgemma", "gpt summary")
        assert isinstance(result, str)


# ============================================================
# 7. GROQ_Prompt — generate_master_consensus_2
# ============================================================
class TestGROQPrompt:
    @pytest.fixture(autouse=True)
    def _reload(self):
        if "GROQ_Prompt" in sys.modules:
            del sys.modules["GROQ_Prompt"]
        import GROQ_Prompt
        self._mod = GROQ_Prompt
        yield

    def test_groq_exception_returns_error_string(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("Connection timeout")
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus_2("llava out", "context", "gpt report")
        assert "Could not generate final consensus" in result

    def test_none_context_does_not_crash(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Final report"
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_completion
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus_2("llava out", None, "gpt report")
        assert isinstance(result, str)

    def test_dict_llava_data_serialised_without_crash(self):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Report"
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_completion
        with patch.object(self._mod, "Groq", return_value=mock_groq):
            result = self._mod.generate_master_consensus_2({"key": "val"}, "context", "gpt")
        assert isinstance(result, str)


# ============================================================
# 8. FastAPI — /v1/analyze  (auth + YOLO guard)
# ============================================================
# We build the TestClient once per test by patching all AI imports at sys.modules level
# so the heavy YOLO model weights are never loaded.

_AI_MODULE_STUBS = {
    "YOLO_Detector":      MagicMock(),
    "LLAVA_FineTuned":    MagicMock(),
    "MedGamma_FineTuned": MagicMock(),
    "GPTNano_Context":    MagicMock(),
    "Groq_Summary":       MagicMock(),
    "GPTNano_Prompt":     MagicMock(),
    "GROQ_Prompt":        MagicMock(),
    "LLAVA_Prompt":       MagicMock(),
}


@pytest.fixture(scope="module")
def app_client():
    """
    Build a FastAPI TestClient with all AI modules stubbed out.
    Scoped to module so the app is only instantiated once.
    """
    from fastapi.testclient import TestClient

    stubs = {k: MagicMock() for k in _AI_MODULE_STUBS}

    with patch.dict(sys.modules, stubs):
        if "main" in sys.modules:
            del sys.modules["main"]
        with patch("sqlite3.connect"):           # Prevent real DB creation
            import main
            importlib.reload(main)
            client = TestClient(main.app, raise_server_exceptions=False)
            yield client, main


class TestAnalyzeEndpoint:
    # ---- Missing API key → 401 -----------------------------------------------
    def test_missing_api_key_returns_401(self, app_client):
        client, _ = app_client
        resp = client.post(
            "/v1/analyze",
            files={"file": ("ecg.png", _make_png_bytes(), "image/png")}
        )
        assert resp.status_code == 401

    # ---- YOLO returns error dict → 400, pipeline stops -----------------------
    def test_yolo_no_ecg_returns_400(self, app_client):
        client, app = app_client
        with patch.object(app, "verify_api_key", return_value=True), \
             patch("main.crop_ecg_from_bytes",
                   return_value={"status": "error", "message": "No ECG detected by YOLO"}):
            resp = client.post(
                "/v1/analyze",
                files={"file": ("ecg.png", _make_png_bytes(), "image/png")},
                headers={"x-api-key": "dasa_12345678901234567890"}
            )
        assert resp.status_code == 400
        assert "No ECG detected" in resp.json()["detail"]


# ============================================================
# 9. FastAPI — /v1/analyze-dynamic  (routing + auth + YOLO guard)
# ============================================================
class TestAnalyzeDynamicEndpoint:
    # ---- Missing API key → 401 -----------------------------------------------
    def test_missing_api_key_returns_401(self, app_client):
        client, _ = app_client
        resp = client.post(
            "/v1/analyze-dynamic",
            data={"prompt": "Is this normal?"}
        )
        assert resp.status_code == 401

    # ---- Text-only: LLaVA must NOT be called ---------------------------------
    def test_text_only_skips_llava(self, app_client):
        client, app = app_client
        llava_spy = MagicMock()
        app.query_llava_2               = llava_spy
        app.generate_clinical_summary_2 = MagicMock(return_value="GPT report")
        app.generate_master_consensus_2 = MagicMock(return_value="Final")

        with patch.object(app, "verify_api_key", return_value=True):
            resp = client.post(
                "/v1/analyze-dynamic",
                data={"prompt": "Is this ECG normal?"},
                headers={"x-api-key": "dasa_12345678901234567890"}
            )
        llava_spy.assert_not_called()
        assert resp.status_code == 200
        assert resp.json()["modality_used"] == "Text Only"

    # ---- YOLO no-detection with image → 400 ---------------------------------
    def test_image_yolo_no_ecg_returns_400(self, app_client):
        client, app = app_client
        with patch.object(app, "verify_api_key", return_value=True), \
             patch.object(app, "crop_ecg_from_bytes",
                          return_value={"status": "error", "message": "No ECG detected by YOLO"}):
            resp = client.post(
                "/v1/analyze-dynamic",
                data={"prompt": "Analyze"},
                files={"file": ("ecg.png", _make_png_bytes(), "image/png")},
                headers={"x-api-key": "dasa_12345678901234567890"}
            )
        assert resp.status_code == 400
        assert "No ECG detected" in resp.json()["detail"]

    # ---- Image present → LLaVA MUST be called --------------------------------
    def test_image_path_calls_llava(self, app_client):
        client, app = app_client
        app.crop_ecg_from_bytes         = MagicMock(return_value=_make_pil_image())
        app.query_llava_2               = MagicMock(return_value={"finding": "AF"})
        app.generate_clinical_summary_2 = MagicMock(return_value="GPT report")
        app.generate_master_consensus_2 = MagicMock(return_value="Final report")

        with patch.object(app, "verify_api_key", return_value=True):
            resp = client.post(
                "/v1/analyze-dynamic",
                data={"prompt": "Analyze ECG"},
                files={"file": ("ecg.png", _make_png_bytes(), "image/png")},
                headers={"x-api-key": "dasa_12345678901234567890"}
            )
        app.query_llava_2.assert_called_once()
        assert resp.status_code == 200
        assert resp.json()["modality_used"] == "Text + Image"

    # ---- prompt + context (no image) → Text Only, LLaVA skipped ------------
    def test_prompt_and_context_skips_llava(self, app_client):
        client, app = app_client
        llava_spy = MagicMock()
        app.query_llava_2               = llava_spy
        app.generate_clinical_summary_2 = MagicMock(return_value="GPT report")
        app.generate_master_consensus_2 = MagicMock(return_value="Final")

        with patch.object(app, "verify_api_key", return_value=True):
            resp = client.post(
                "/v1/analyze-dynamic",
                data={"prompt": "Any concerns?", "context": "Patient is 55, male"},
                headers={"x-api-key": "dasa_12345678901234567890"}
            )
        llava_spy.assert_not_called()
        assert resp.status_code == 200
        assert resp.json()["modality_used"] == "Text Only"
