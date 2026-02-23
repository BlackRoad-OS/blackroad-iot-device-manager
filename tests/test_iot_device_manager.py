"""
Tests for IoTDeviceManager
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from iot_device_manager import IoTDeviceManager  # noqa: E402


@pytest.fixture
def mgr(tmp_path):
    db_path = str(tmp_path / "test_iot.db")
    manager = IoTDeviceManager(db_path=db_path)
    yield manager
    manager.close()


def test_register_device(mgr):
    device = mgr.register_device(
        name="sensor-001",
        device_type="temperature",
        firmware_version="2.1.0",
        ip_address="192.168.1.10",
    )
    assert device.id is not None
    assert device.name == "sensor-001"
    assert device.type == "temperature"
    assert device.firmware_version == "2.1.0"
    assert device.ip_address == "192.168.1.10"
    assert device.status == "online"

    row = mgr.conn.execute(
        "SELECT * FROM devices WHERE id=?", (device.id,)
    ).fetchone()
    assert row is not None
    assert row["name"] == "sensor-001"


def test_ingest_telemetry(mgr):
    device = mgr.register_device("cam-001", "camera", ip_address="10.0.0.1")
    metrics = {
        "temperature": (72.5, "F"),
        "humidity": (45.0, "%"),
        "voltage": (3.3, "V"),
    }
    alerts = mgr.ingest_telemetry(device.id, metrics)
    assert isinstance(alerts, list)

    count = mgr.conn.execute(
        "SELECT COUNT(*) FROM telemetry WHERE device_id=?", (device.id,)
    ).fetchone()[0]
    assert count == 3


def test_alert_rule_trigger(mgr):
    device = mgr.register_device("pump-001", "actuator", ip_address="10.0.0.2")
    mgr.add_alert_rule(
        device_id=device.id,
        metric_name="pressure",
        operator=">",
        threshold=100.0,
        severity="critical",
    )
    # Value below threshold — no alert
    alerts_low = mgr.ingest_telemetry(device.id, {"pressure": (50.0, "psi")})
    assert len(alerts_low) == 0

    # Value above threshold — alert must fire
    alerts_high = mgr.ingest_telemetry(device.id, {"pressure": (150.0, "psi")})
    assert len(alerts_high) == 1
    assert "critical" in alerts_high[0].lower()
    assert "pressure" in alerts_high[0]


def test_firmware_queue(mgr):
    device = mgr.register_device("router-001", "gateway", ip_address="10.0.0.3")
    fw = mgr.queue_firmware_update(device.id, "3.0.0", checksum="abc123")
    assert fw.id is not None
    assert fw.status == "pending"

    row = mgr.conn.execute(
        "SELECT status FROM firmware_queue WHERE id=?", (fw.id,)
    ).fetchone()
    assert row["status"] == "pending"

    processed = mgr.process_firmware_queue(limit=5)
    assert fw.id in processed

    row_after = mgr.conn.execute(
        "SELECT status FROM firmware_queue WHERE id=?", (fw.id,)
    ).fetchone()
    assert row_after["status"] in ("completed", "failed")


def test_device_group(mgr):
    group = mgr.create_group("factory-floor", "Production floor sensors")
    assert group.id is not None
    assert group.name == "factory-floor"

    d1 = mgr.register_device("dev-a", "sensor", ip_address="10.1.0.1")
    d2 = mgr.register_device("dev-b", "sensor", ip_address="10.1.0.2")
    mgr.add_device_to_group(d1.id, group.id)
    mgr.add_device_to_group(d2.id, group.id)

    members = mgr.conn.execute(
        "SELECT COUNT(*) FROM group_members WHERE group_id=?", (group.id,)
    ).fetchone()[0]
    assert members == 2


def test_fleet_dashboard(mgr):
    for i in range(4):
        mgr.register_device(f"fleet-dev-{i}", "sensor", ip_address=f"10.2.0.{i}")

    dash = mgr.get_fleet_dashboard()
    assert dash["total_devices"] >= 4
    assert dash["online_count"] >= 4
    assert dash["offline_count"] >= 0
    assert "pending_firmware" in dash
    assert "active_alerts" in dash
    assert "groups" in dash
