"""Tests for the app's HTTP surface.

A fake pipeline stands in for the models: these tests are about what the
server does with a reading, not about inference, and loading 190 MB of weights
per test would make them useless to run.
"""

import csv
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

from src.app.server import create_app
from src.physics.deterioration import assess
from src.pipeline import Detections, LotReading


class FakePipeline:
    """Returns a damage percentage taken from the lot id, so tests can order lots."""

    def __init__(self):
        self.calls = []

    def analyse(self, image_path, *, lot_id, temperature_c, moisture_pct_wb):
        self.calls.append(lot_id)
        damage = float(lot_id.split("_")[-1])
        reading = LotReading(
            lot_id=lot_id,
            kernels_counted=200,
            damage_mass_pct=damage,
            damage_count_pct=damage,
            biological_pct={"mould_suspect": 1.0},
            class_counts={"sound": 190, "fragment": 10},
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
            assessment=assess(damage, temperature_c, moisture_pct_wb),
        )
        # Instances are sized from the image, as the real pipeline does: it
        # segments the array it was handed, so the two geometries cannot
        # disagree. A fake that returns a fixed shape would let an overlay bug
        # through by making draw() tolerate a mismatch that never happens.
        width, height = Image.open(image_path).size
        instances = np.zeros((height, width), dtype=int)
        instances[2:5, 2:5] = 1
        return reading, Detections(
            instances=instances, labels=[1], predictions=["fragment"]
        )


class FailingPipeline:
    def analyse(self, image_path, **kwargs):
        raise ValueError(f"no kernels found in {image_path}")


@pytest.fixture
def client():
    return TestClient(create_app(pipeline_factory=FakePipeline))


def _photo(width=40, height=30):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client, lot_id, temperature=20.0, moisture=14.0):
    return client.post(
        "/api/lots",
        files={"photo": (f"{lot_id}.png", _photo(), "image/png")},
        data={
            "lot_id": lot_id,
            "temperature_c": str(temperature),
            "moisture_pct_wb": str(moisture),
        },
    )


def test_a_photograph_produces_a_reading():
    client = TestClient(create_app(pipeline_factory=FakePipeline))
    response = _upload(client, "lot_5.0")
    assert response.status_code == 201
    body = response.json()
    assert body["lot_id"] == "lot_5.0"
    assert body["damage_mass_pct"] == pytest.approx(5.0)
    assert body["kernels_counted"] == 200


def test_a_reading_carries_its_resolution_limit(client):
    # Without this the interface can show an ordering the sample cannot support.
    body = _upload(client, "lot_5.0").json()
    assert body["resolvable_gap_pct"] > 0.0


def test_a_reading_carries_absolute_days_and_their_error(client):
    body = _upload(client, "lot_5.0").json()
    assert body["mode"] == "ranking+absolute"
    assert body["days_to_threshold"] > 0.0
    assert body["days_to_threshold_error_pct"] > 0.0
    # Nothing was withheld, so nothing may claim to have been.
    assert body["suppression_reasons"] == []
    # The standing caveats still travel with the number.
    assert body["model_notes"]


def test_reading_exposes_localizable_model_message_descriptors(client):
    body = _upload(client, "lot_5.0").json()
    assert body["model_messages"]
    assert all(
        {"key", "values", "fallback"} <= set(item)
        for item in body["model_messages"]
    )


def test_a_reading_says_whether_it_was_corrected(client):
    body = _upload(client, "lot_5.0").json()
    assert body["calibrated"] is False
    assert body["raw_damage_mass_pct"] == pytest.approx(body["damage_mass_pct"])


def test_lots_come_back_ranked_fastest_first(client):
    _upload(client, "lot_3.0")
    _upload(client, "lot_30.0")
    ranking = client.get("/api/lots").json()["ranking"]
    assert [entry["lot_id"] for entry in ranking] == ["lot_30.0", "lot_3.0"]
    assert ranking[0]["rank"] == 1


def test_lots_the_measurement_cannot_separate_are_reported_as_tied(client):
    # At 200 kernels a one-point gap at 5% damage is inside the sampling noise.
    _upload(client, "lot_5.0")
    _upload(client, "lot_6.0")
    ranking = client.get("/api/lots").json()["ranking"]
    assert ranking[0]["tied_with"] == ["lot_5.0"]


def test_rephotographing_a_lot_replaces_its_reading(client):
    _upload(client, "lot_5.0")
    _upload(client, "lot_5.0")
    assert len(client.get("/api/lots").json()["lots"]) == 1


def test_a_lot_can_be_deleted(client):
    _upload(client, "lot_5.0")
    assert client.delete("/api/lots/lot_5.0").status_code == 204
    assert client.get("/api/lots").json()["lots"] == []


def test_deleting_a_lot_that_was_never_there_is_a_404(client):
    assert client.delete("/api/lots/nobody").status_code == 404


def test_resetting_empties_the_session(client):
    _upload(client, "lot_5.0")
    assert client.post("/api/session/reset").status_code == 204
    assert client.get("/api/lots").json()["lots"] == []


def test_an_upload_that_is_not_an_image_is_refused(client):
    response = client.post(
        "/api/lots",
        files={"photo": ("notes.txt", b"not a photograph", "text/plain")},
        data={"lot_id": "lot_5.0", "temperature_c": "20", "moisture_pct_wb": "14"},
    )
    assert response.status_code == 415


def test_a_photograph_with_no_kernels_says_so_and_leaves_the_session_alone():
    client = TestClient(create_app(pipeline_factory=FailingPipeline))
    response = _upload(client, "lot_5.0")
    assert response.status_code == 422
    assert "no kernels" in response.json()["detail"]
    assert client.get("/api/lots").json()["lots"] == []


def test_the_overlay_is_served_for_a_known_lot(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/lots/lot_5.0/overlay.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (40, 30)


def test_the_overlay_of_an_unknown_lot_is_a_404(client):
    assert client.get("/api/lots/nobody/overlay.png").status_code == 404


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'href="/api/report.pdf"' in response.text
    assert 'href="/api/report.csv"' in response.text


def test_the_page_exposes_the_measurement_flow_to_assistive_technology(client):
    response = client.get("/")

    assert 'aria-label="Measurement flow"' in response.text
    assert "Capture sample" in response.text
    assert "Measure kernels" in response.text
    assert "Rank lots" in response.text


def test_the_models_are_loaded_once_not_per_photograph():
    # Model load is 1.2 s. Per-request loading would make the app unusable and
    # would be invisible in a test that only checks the numbers.
    created = []

    def factory():
        created.append(1)
        return FakePipeline()

    client = TestClient(create_app(pipeline_factory=factory))
    _upload(client, "lot_5.0")
    _upload(client, "lot_9.0")
    assert len(created) == 1


# --- Session isolation -----------------------------------------------------
# The store was process-wide. Published, that puts every visitor into one
# ranking. These pin the boundary at the HTTP layer, where the cookie is.

def _app():
    return create_app(pipeline_factory=FakePipeline)


def test_two_visitors_do_not_see_each_other_s_lots():
    app = _app()
    alice, bob = TestClient(app), TestClient(app)

    _upload(alice, "lot_5.0")
    _upload(bob, "lot_9.0")

    assert [r["lot_id"] for r in alice.get("/api/lots").json()["lots"]] == ["lot_5.0"]
    assert [r["lot_id"] for r in bob.get("/api/lots").json()["lots"]] == ["lot_9.0"]


def test_a_visitor_is_issued_a_session_cookie():
    response = TestClient(_app()).get("/api/lots")
    assert "session_id" in response.cookies


def test_the_session_cookie_is_not_readable_by_script():
    # The id is a bearer token for a session; script access would let any
    # injected content on the page hand it to somebody else.
    response = TestClient(_app()).get("/api/lots")
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header


def test_a_forged_cookie_lands_in_a_fresh_session_not_someone_else_s():
    app = _app()
    alice = TestClient(app)
    _upload(alice, "lot_5.0")

    intruder = TestClient(app)
    intruder.cookies.set("session_id", "guessed")
    assert intruder.get("/api/lots").json()["lots"] == []


def test_one_visitor_s_reset_leaves_another_s_lots_alone():
    app = _app()
    alice, bob = TestClient(app), TestClient(app)
    _upload(alice, "lot_5.0")
    _upload(bob, "lot_9.0")

    alice.post("/api/session/reset")

    assert alice.get("/api/lots").json()["lots"] == []
    assert len(bob.get("/api/lots").json()["lots"]) == 1


# --- Downloadable reports -------------------------------------------------

def test_csv_report_downloads_ranked_lot_evidence(client):
    _upload(client, "lot_3.0", temperature=19.0, moisture=13.0)
    _upload(client, "lot_30.0", temperature=27.0, moisture=17.0)

    response = client.get("/api/report.csv")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == \
        'attachment; filename="maize-lot-report.csv"'
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert [row["lot_id"] for row in rows] == ["lot_30.0", "lot_3.0"]
    assert rows[0]["rank"] == "1"
    assert rows[0]["mechanical_damage_mass_pct"] == "30.00"
    assert rows[0]["temperature_c"] == "27.00"
    assert rows[0]["moisture_pct_wb"] == "17.00"
    assert rows[0]["kernels_counted"] == "200"
    assert float(rows[0]["days_to_threshold"]) > 0.0
    assert rows[0]["model_warnings"]


def test_csv_report_neutralises_spreadsheet_formulas_in_lot_names(client):
    _upload(client, "=2+2_5.0")

    response = client.get("/api/report.csv")
    row = next(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))

    assert row["lot_id"] == "'=2+2_5.0"


@pytest.mark.parametrize(
    ("lang", "rank_header", "yes_value", "suffix"),
    [
        ("hi", "क्रम", "नहीं", "-hi.csv"),
        ("te", "ర్యాంక్", "కాదు", "-te.csv"),
    ],
)
def test_csv_report_uses_requested_language(client, lang, rank_header, yes_value, suffix):
    _upload(client, "lot_5.0")

    response = client.get(f"/api/report.csv?lang={lang}")
    text = response.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert response.content.startswith(b"\xef\xbb\xbf")
    assert rank_header in rows[0]
    assert rows[0][rank_header] == "1"
    assert yes_value in rows[0].values()
    assert rows[0]["लॉट_नाम" if lang == "hi" else "లాట్_పేరు"] == "lot_5.0"
    assert suffix in response.headers["content-disposition"]


def test_invalid_csv_language_falls_back_to_existing_english_contract(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/report.csv?lang=fr")
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert rows[0]["rank"] == "1"
    assert 'filename="maize-lot-report.csv"' in response.headers["content-disposition"]


def test_pdf_report_downloads_the_same_session_evidence(client):
    _upload(client, "lot_5.0")

    response = client.get("/api/report.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == \
        'attachment; filename="maize-lot-report.pdf"'
    assert response.content.startswith(b"%PDF-")
    assert b"lot_5.0" in response.content
    assert b"Mechanical damage" in response.content
    assert b"Note:" not in response.content
    assert b"Steele (1967) estimate" not in response.content


@pytest.mark.parametrize(
    ("lang", "title", "suffix"),
    [
        ("hi", "मक्का लॉट रिपोर्ट", "-hi.pdf"),
        ("te", "మొక్కజొన్న లాట్ నివేదిక", "-te.pdf"),
    ],
)
def test_pdf_report_uses_requested_language_and_embedded_unicode_font(
    client, lang, title, suffix
):
    _upload(client, "lot_5.0")
    response = client.get(f"/api/report.pdf?lang={lang}")
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(io.BytesIO(response.content)).pages
    )

    assert response.content.startswith(b"%PDF-")
    assert title in text
    assert "lot_5.0" in text
    assert "Note:" not in text
    assert suffix in response.headers["content-disposition"]


def test_invalid_pdf_language_falls_back_to_english(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/report.pdf?lang=fr")
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(io.BytesIO(response.content)).pages
    )
    assert "Maize lot report" in text
    assert 'filename="maize-lot-report.pdf"' in response.headers[
        "content-disposition"
    ]


@pytest.mark.parametrize("path", ["/api/report.csv", "/api/report.pdf"])
def test_an_empty_session_has_no_report_to_download(client, path):
    response = client.get(path)

    assert response.status_code == 409
    assert response.json()["detail"] == "No lots to export."


def test_a_visitor_s_report_never_contains_another_visitor_s_lots():
    app = _app()
    alice, bob = TestClient(app), TestClient(app)
    _upload(alice, "lot_5.0")
    _upload(bob, "lot_9.0")

    alice_report = alice.get("/api/report.csv").text
    bob_report = bob.get("/api/report.csv").text

    assert "lot_5.0" in alice_report
    assert "lot_9.0" not in alice_report
    assert "lot_9.0" in bob_report
    assert "lot_5.0" not in bob_report


def test_a_visitor_cannot_delete_another_visitor_s_lot():
    app = _app()
    alice, bob = TestClient(app), TestClient(app)
    _upload(alice, "lot_5.0")
    bob.get("/api/lots")

    assert bob.delete("/api/lots/lot_5.0").status_code == 404
    assert len(alice.get("/api/lots").json()["lots"]) == 1


def test_a_visitor_cannot_read_another_visitor_s_overlay():
    # Overlays were a process-wide dict keyed by lot id alone, so two visitors
    # using the same lot name would have served each other their photographs.
    app = _app()
    alice, bob = TestClient(app), TestClient(app)
    _upload(alice, "lot_5.0")
    bob.get("/api/lots")

    assert bob.get("/api/lots/lot_5.0/overlay.png").status_code == 404
    assert alice.get("/api/lots/lot_5.0/overlay.png").status_code == 200


def test_the_same_visitor_keeps_their_lots_across_requests():
    client = TestClient(_app())
    _upload(client, "lot_5.0")
    _upload(client, "lot_9.0")
    assert len(client.get("/api/lots").json()["lots"]) == 2


# --- Guards for a public URL -----------------------------------------------
# Each of these costs the server an inference, so they are the endpoints that
# have to be defended before the app is reachable from outside the network.

def test_an_oversized_photograph_is_refused_before_inference():
    from src.app.limits import MAX_UPLOAD_BYTES

    pipeline = FakePipeline()
    client = TestClient(create_app(pipeline_factory=lambda: pipeline))
    oversized = b"\x89PNG\r\n\x1a\n" + b"\0" * (MAX_UPLOAD_BYTES + 1024)

    response = client.post(
        "/api/lots",
        files={"photo": ("huge.png", oversized, "image/png")},
        data={"lot_id": "lot_5.0", "temperature_c": "20", "moisture_pct_wb": "14"},
    )
    assert response.status_code == 413
    # The point of the cap is that the models are never reached.
    assert pipeline.calls == []


def test_repeated_measurements_are_rate_limited():
    from src.app.limits import RateLimiter

    client = TestClient(
        create_app(
            pipeline_factory=FakePipeline,
            limiter=RateLimiter(limit=2, window_s=60),
        )
    )
    assert _upload(client, "lot_1.0").status_code == 201
    assert _upload(client, "lot_2.0").status_code == 201

    refused = _upload(client, "lot_3.0")
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0


def test_a_rate_limited_caller_keeps_the_lots_they_already_measured():
    from src.app.limits import RateLimiter

    client = TestClient(
        create_app(pipeline_factory=FakePipeline, limiter=RateLimiter(limit=1, window_s=60))
    )
    _upload(client, "lot_5.0")
    _upload(client, "lot_9.0")

    assert [r["lot_id"] for r in client.get("/api/lots").json()["lots"]] == ["lot_5.0"]


def test_reading_the_ranking_is_not_rate_limited():
    # Only inference is expensive. Throttling reads would punish the page for
    # refreshing itself.
    from src.app.limits import RateLimiter

    client = TestClient(
        create_app(pipeline_factory=FakePipeline, limiter=RateLimiter(limit=1, window_s=60))
    )
    for _ in range(10):
        assert client.get("/api/lots").status_code == 200


# --- The page's own wiring -------------------------------------------------

def test_the_visible_upload_surface_actually_opens_the_file_picker(client):
    # The dropzone is the only large target on the capture screen and the file
    # input behind it is visually hidden. If the dropzone is not a label bound
    # to that input, clicking it does nothing at all -- and nothing else in the
    # test suite exercises a browser, so this is the only place that catches it.
    page = client.get("/").text
    assert 'class="dropzone"' in page
    dropzone = page[page.index('class="dropzone"') - 40 : page.index('class="dropzone"') + 120]
    assert dropzone.lstrip().startswith("<label") or "<label" in dropzone
    assert 'for="f-photo"' in dropzone
