"""Tests for the app's HTTP surface.

A fake pipeline stands in for the models: these tests are about what the
server does with a reading, not about inference, and loading 190 MB of weights
per test would make them useless to run.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


def test_a_reading_says_absolute_days_are_gated(client):
    body = _upload(client, "lot_5.0").json()
    assert body["mode"] == "ranking"
    assert body["days_to_threshold"] is None
    assert body["suppression_reasons"]


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
