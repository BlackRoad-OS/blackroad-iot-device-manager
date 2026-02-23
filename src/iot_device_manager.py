"""
BlackRoad IoT Device Manager
Comprehensive IoT device lifecycle management system.
"""
import dataclasses
import sqlite3
import datetime
import json
import argparse
import sys
import os
import random
import math
from typing import List, Dict, Optional, Tuple, Any

# ANSI color constants
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BLUE = "\033[0;34m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclasses.dataclass
class Device:
    id: str
    name: str
    type: str
    group_id: Optional[str]
    firmware_version: str
    status: str
    ip_address: str
    last_seen: str
    metadata: Dict[str, Any]


@dataclasses.dataclass
class TelemetryRecord:
    device_id: str
    timestamp: str
    metric_name: str
    value: float
    unit: str


@dataclasses.dataclass
class AlertRule:
    id: str
    device_id: str
    metric_name: str
    operator: str
    threshold: float
    severity: str
    enabled: bool


@dataclasses.dataclass
class FirmwareUpdate:
    id: str
    device_id: str
    version: str
    status: str
    queued_at: str
    completed_at: Optional[str]
    checksum: str


@dataclasses.dataclass
class DeviceGroup:
    id: str
    name: str
    description: str
    created_at: str


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


def _uid() -> str:
    import uuid
    return str(uuid.uuid4())


class IoTDeviceManager:
    """Core IoT device management engine backed by SQLite."""

    def __init__(self, db_path: str = "iot_devices.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_db()

    def init_db(self) -> None:
        """Create all required tables."""
        ddl = """
        CREATE TABLE IF NOT EXISTS device_groups (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            group_id TEXT REFERENCES device_groups(id),
            firmware_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'online',
            ip_address TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL REFERENCES devices(id),
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS alert_rules (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL REFERENCES devices(id),
            metric_name TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold REAL NOT NULL,
            severity TEXT NOT NULL DEFAULT 'warning',
            enabled INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS firmware_queue (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL REFERENCES devices(id),
            version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            queued_at TEXT NOT NULL,
            completed_at TEXT,
            checksum TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS group_members (
            device_id TEXT NOT NULL REFERENCES devices(id),
            group_id TEXT NOT NULL REFERENCES device_groups(id),
            PRIMARY KEY (device_id, group_id)
        );
        """
        self.conn.executescript(ddl)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def register_device(
        self,
        name: str,
        device_type: str,
        group_id: Optional[str] = None,
        firmware_version: str = "1.0.0",
        ip_address: str = "0.0.0.0",
    ) -> Device:
        """Register a new device and return its dataclass representation."""
        device = Device(
            id=_uid(),
            name=name,
            type=device_type,
            group_id=group_id,
            firmware_version=firmware_version,
            status="online",
            ip_address=ip_address,
            last_seen=_now(),
            metadata={},
        )
        self.conn.execute(
            """INSERT INTO devices
               (id, name, type, group_id, firmware_version, status, ip_address, last_seen, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device.id, device.name, device.type, device.group_id,
                device.firmware_version, device.status, device.ip_address,
                device.last_seen, json.dumps(device.metadata),
            ),
        )
        self.conn.commit()
        return device

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def ingest_telemetry(self, device_id: str, metrics: Dict[str, Tuple[float, str]]) -> List[str]:
        """
        Bulk-insert telemetry records and evaluate alert rules.

        metrics format: {metric_name: (value, unit)}
        Returns list of triggered alert descriptions.
        """
        ts = _now()
        rows = [
            (device_id, ts, name, val, unit)
            for name, (val, unit) in metrics.items()
        ]
        self.conn.executemany(
            "INSERT INTO telemetry (device_id, timestamp, metric_name, value, unit) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.execute(
            "UPDATE devices SET last_seen=? WHERE id=?", (ts, device_id)
        )
        self.conn.commit()

        triggered: List[str] = []
        for metric_name, (value, _unit) in metrics.items():
            triggered.extend(self._evaluate_alerts(device_id, metric_name, value))
        return triggered

    def _evaluate_alerts(self, device_id: str, metric_name: str, value: float) -> List[str]:
        cursor = self.conn.execute(
            """SELECT * FROM alert_rules
               WHERE device_id=? AND metric_name=? AND enabled=1""",
            (device_id, metric_name),
        )
        fired: List[str] = []
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        for row in cursor.fetchall():
            op_fn = ops.get(row["operator"])
            if op_fn and op_fn(value, row["threshold"]):
                msg = (
                    f"[{row['severity'].upper()}] Device {device_id}: "
                    f"{metric_name} {row['operator']} {row['threshold']} "
                    f"(current={value})"
                )
                fired.append(msg)
        return fired

    # ------------------------------------------------------------------
    # Alert rules
    # ------------------------------------------------------------------

    def add_alert_rule(
        self,
        device_id: str,
        metric_name: str,
        operator: str,
        threshold: float,
        severity: str = "warning",
    ) -> AlertRule:
        rule = AlertRule(
            id=_uid(),
            device_id=device_id,
            metric_name=metric_name,
            operator=operator,
            threshold=threshold,
            severity=severity,
            enabled=True,
        )
        self.conn.execute(
            """INSERT INTO alert_rules
               (id, device_id, metric_name, operator, threshold, severity, enabled)
               VALUES (?,?,?,?,?,?,?)""",
            (rule.id, rule.device_id, rule.metric_name, rule.operator,
             rule.threshold, rule.severity, int(rule.enabled)),
        )
        self.conn.commit()
        return rule

    # ------------------------------------------------------------------
    # Firmware
    # ------------------------------------------------------------------

    def queue_firmware_update(
        self, device_id: str, version: str, checksum: str
    ) -> FirmwareUpdate:
        update = FirmwareUpdate(
            id=_uid(),
            device_id=device_id,
            version=version,
            status="pending",
            queued_at=_now(),
            completed_at=None,
            checksum=checksum,
        )
        self.conn.execute(
            """INSERT INTO firmware_queue
               (id, device_id, version, status, queued_at, completed_at, checksum)
               VALUES (?,?,?,?,?,?,?)""",
            (update.id, update.device_id, update.version, update.status,
             update.queued_at, update.completed_at, update.checksum),
        )
        self.conn.commit()
        return update

    def process_firmware_queue(self, limit: int = 5) -> List[str]:
        """Simulate processing pending firmware updates. Returns processed job IDs."""
        cursor = self.conn.execute(
            "SELECT * FROM firmware_queue WHERE status='pending' LIMIT ?", (limit,)
        )
        jobs = cursor.fetchall()
        processed: List[str] = []
        for job in jobs:
            # Simulate 90% success rate
            success = random.random() > 0.1
            new_status = "completed" if success else "failed"
            completed_at = _now()
            self.conn.execute(
                "UPDATE firmware_queue SET status=?, completed_at=? WHERE id=?",
                (new_status, completed_at, job["id"]),
            )
            if success:
                self.conn.execute(
                    "UPDATE devices SET firmware_version=? WHERE id=?",
                    (job["version"], job["device_id"]),
                )
            processed.append(job["id"])
        self.conn.commit()
        return processed

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def create_group(self, name: str, description: str = "") -> DeviceGroup:
        group = DeviceGroup(id=_uid(), name=name, description=description, created_at=_now())
        self.conn.execute(
            "INSERT INTO device_groups (id, name, description, created_at) VALUES (?,?,?,?)",
            (group.id, group.name, group.description, group.created_at),
        )
        self.conn.commit()
        return group

    def add_device_to_group(self, device_id: str, group_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO group_members (device_id, group_id) VALUES (?,?)",
            (device_id, group_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """Return telemetry summary: avg/min/max per metric plus last-24h count."""
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=24)).isoformat()
        cursor = self.conn.execute(
            """SELECT metric_name,
                      AVG(value) as avg_val,
                      MIN(value) as min_val,
                      MAX(value) as max_val,
                      COUNT(*) as cnt
               FROM telemetry
               WHERE device_id=? AND timestamp >= ?
               GROUP BY metric_name""",
            (device_id, cutoff),
        )
        metrics: Dict[str, Any] = {}
        for row in cursor.fetchall():
            metrics[row["metric_name"]] = {
                "avg": round(row["avg_val"], 4),
                "min": round(row["min_val"], 4),
                "max": round(row["max_val"], 4),
                "count_24h": row["cnt"],
            }
        return {"device_id": device_id, "metrics": metrics}

    def get_fleet_dashboard(self) -> Dict[str, Any]:
        """Return high-level fleet metrics."""
        total = self.conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
        online = self.conn.execute(
            "SELECT COUNT(*) FROM devices WHERE status='online'"
        ).fetchone()[0]
        offline = total - online
        pending_fw = self.conn.execute(
            "SELECT COUNT(*) FROM firmware_queue WHERE status='pending'"
        ).fetchone()[0]
        active_alerts = self.conn.execute(
            "SELECT COUNT(*) FROM alert_rules WHERE enabled=1"
        ).fetchone()[0]
        groups_count = self.conn.execute("SELECT COUNT(*) FROM device_groups").fetchone()[0]
        return {
            "total_devices": total,
            "online_count": online,
            "offline_count": offline,
            "pending_firmware": pending_fw,
            "active_alerts": active_alerts,
            "groups": groups_count,
        }

    def get_telemetry_trend(
        self, device_id: str, metric_name: str, hours: int = 24
    ) -> List[Tuple[str, float, float]]:
        """
        Return (timestamp, value, moving_avg) tuples for the given metric.
        Moving average uses a window of 5.
        """
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()
        cursor = self.conn.execute(
            """SELECT timestamp, value FROM telemetry
               WHERE device_id=? AND metric_name=? AND timestamp>=?
               ORDER BY timestamp ASC""",
            (device_id, metric_name, cutoff),
        )
        rows = cursor.fetchall()
        window = 5
        result: List[Tuple[str, float, float]] = []
        values = [r["value"] for r in rows]
        for i, row in enumerate(rows):
            start = max(0, i - window + 1)
            ma = sum(values[start: i + 1]) / (i - start + 1)
            result.append((row["timestamp"], row["value"], round(ma, 4)))
        return result

    def detect_anomalies(
        self, device_id: str, metric_name: str, z_score_threshold: float = 2.5
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using z-score from last 100 readings."""
        cursor = self.conn.execute(
            """SELECT timestamp, value FROM telemetry
               WHERE device_id=? AND metric_name=?
               ORDER BY timestamp DESC LIMIT 100""",
            (device_id, metric_name),
        )
        rows = cursor.fetchall()
        if len(rows) < 3:
            return []
        values = [r["value"] for r in rows]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance > 0 else 0
        if std == 0:
            return []
        anomalies: List[Dict[str, Any]] = []
        for row in rows:
            z = abs(row["value"] - mean) / std
            if z >= z_score_threshold:
                anomalies.append({
                    "timestamp": row["timestamp"],
                    "value": row["value"],
                    "z_score": round(z, 3),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                })
        return anomalies

    def close(self) -> None:
        self.conn.close()


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def _header(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}\n")


def _row(label: str, value: Any, color: str = RESET) -> None:
    print(f"  {BOLD}{label:<25}{RESET} {color}{value}{RESET}")


def _table(headers: List[str], rows: List[List[Any]]) -> None:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(f"{BOLD}{CYAN}" + fmt.format(*headers) + RESET)
    print(f"{CYAN}{sep}{RESET}")
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ------------------------------------------------------------------
# CLI subcommands
# ------------------------------------------------------------------

def cmd_register(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    device = mgr.register_device(
        name=args.name,
        device_type=args.type,
        group_id=args.group_id,
        firmware_version=args.firmware,
        ip_address=args.ip,
    )
    _header("Device Registered")
    _row("ID", device.id, GREEN)
    _row("Name", device.name, CYAN)
    _row("Type", device.type, YELLOW)
    _row("Firmware", device.firmware_version, MAGENTA)
    _row("IP", device.ip_address)
    _row("Status", device.status, GREEN)
    print()


def cmd_telemetry(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    metrics: Dict[str, Tuple[float, str]] = {}
    for entry in args.metrics:
        parts = entry.split(":")
        if len(parts) < 2:
            print(f"{RED}Invalid metric format '{entry}'. Use name:value[:unit]{RESET}")
            sys.exit(1)
        m_name = parts[0]
        m_val = float(parts[1])
        m_unit = parts[2] if len(parts) > 2 else ""
        metrics[m_name] = (m_val, m_unit)
    alerts = mgr.ingest_telemetry(args.device_id, metrics)
    _header("Telemetry Ingested")
    _row("Device", args.device_id, CYAN)
    _row("Metrics written", len(metrics), GREEN)
    if alerts:
        print(f"\n{BOLD}{RED}  Alerts Triggered:{RESET}")
        for a in alerts:
            print(f"  {RED}⚠  {a}{RESET}")
    else:
        print(f"  {GREEN}✓  No alerts triggered{RESET}")
    print()


def cmd_alert(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    rule = mgr.add_alert_rule(
        device_id=args.device_id,
        metric_name=args.metric,
        operator=args.operator,
        threshold=args.threshold,
        severity=args.severity,
    )
    _header("Alert Rule Added")
    _row("Rule ID", rule.id, GREEN)
    _row("Device", rule.device_id, CYAN)
    _row("Condition", f"{rule.metric_name} {rule.operator} {rule.threshold}", YELLOW)
    _row("Severity", rule.severity, RED if rule.severity == "critical" else YELLOW)
    print()


def cmd_firmware(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    if args.firmware_cmd == "queue":
        fw = mgr.queue_firmware_update(args.device_id, args.version, args.checksum)
        _header("Firmware Update Queued")
        _row("Job ID", fw.id, GREEN)
        _row("Device", fw.device_id, CYAN)
        _row("Version", fw.version, MAGENTA)
        _row("Checksum", fw.checksum)
    elif args.firmware_cmd == "process":
        processed = mgr.process_firmware_queue(limit=args.limit)
        _header("Firmware Queue Processed")
        _row("Jobs processed", len(processed), GREEN)
        for jid in processed:
            print(f"  {CYAN}• {jid}{RESET}")
    print()


def cmd_group(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    if args.group_cmd == "create":
        grp = mgr.create_group(args.name, args.description)
        _header("Group Created")
        _row("Group ID", grp.id, GREEN)
        _row("Name", grp.name, CYAN)
        _row("Description", grp.description)
    elif args.group_cmd == "add":
        mgr.add_device_to_group(args.device_id, args.group_id)
        _header("Device Added to Group")
        _row("Device", args.device_id, CYAN)
        _row("Group", args.group_id, GREEN)
    print()


def cmd_dashboard(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    dash = mgr.get_fleet_dashboard()
    _header("Fleet Dashboard")
    _row("Total Devices", dash["total_devices"], BOLD)
    _row("Online", dash["online_count"], GREEN)
    _row("Offline", dash["offline_count"], RED if dash["offline_count"] > 0 else RESET)
    _row("Pending Firmware", dash["pending_firmware"], YELLOW)
    _row("Active Alert Rules", dash["active_alerts"], CYAN)
    _row("Groups", dash["groups"], MAGENTA)
    print()


def cmd_stats(args: argparse.Namespace, mgr: IoTDeviceManager) -> None:
    stats = mgr.get_device_stats(args.device_id)
    _header(f"Device Stats: {args.device_id}")
    if not stats["metrics"]:
        print(f"  {YELLOW}No telemetry data found in last 24 hours.{RESET}\n")
        return
    headers = ["Metric", "Avg", "Min", "Max", "Count(24h)"]
    rows = [
        [name, d["avg"], d["min"], d["max"], d["count_24h"]]
        for name, d in stats["metrics"].items()
    ]
    _table(headers, rows)
    print()


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iot_device_manager",
        description=f"{BOLD}{CYAN}BlackRoad IoT Device Manager{RESET}",
    )
    parser.add_argument("--db", default="iot_devices.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="command")

    # register
    p_reg = sub.add_parser("register", help="Register a new device")
    p_reg.add_argument("name")
    p_reg.add_argument("--type", default="sensor")
    p_reg.add_argument("--group-id", default=None)
    p_reg.add_argument("--firmware", default="1.0.0")
    p_reg.add_argument("--ip", default="0.0.0.0")

    # telemetry
    p_tel = sub.add_parser("telemetry", help="Ingest telemetry metrics")
    p_tel.add_argument("device_id")
    p_tel.add_argument("metrics", nargs="+", help="name:value[:unit]")

    # alert
    p_ale = sub.add_parser("alert", help="Add alert rule")
    p_ale.add_argument("device_id")
    p_ale.add_argument("metric")
    p_ale.add_argument("operator", choices=[">", "<", ">=", "<=", "=="])
    p_ale.add_argument("threshold", type=float)
    p_ale.add_argument("--severity", default="warning", choices=["info", "warning", "critical"])

    # firmware
    p_fw = sub.add_parser("firmware", help="Firmware update management")
    fw_sub = p_fw.add_subparsers(dest="firmware_cmd")
    p_fw_q = fw_sub.add_parser("queue", help="Queue firmware update")
    p_fw_q.add_argument("device_id")
    p_fw_q.add_argument("version")
    p_fw_q.add_argument("checksum")
    p_fw_p = fw_sub.add_parser("process", help="Process firmware queue")
    p_fw_p.add_argument("--limit", type=int, default=5)

    # group
    p_grp = sub.add_parser("group", help="Device group management")
    grp_sub = p_grp.add_subparsers(dest="group_cmd")
    p_grp_c = grp_sub.add_parser("create", help="Create a device group")
    p_grp_c.add_argument("name")
    p_grp_c.add_argument("--description", default="")
    p_grp_a = grp_sub.add_parser("add", help="Add device to group")
    p_grp_a.add_argument("device_id")
    p_grp_a.add_argument("group_id")

    # dashboard
    sub.add_parser("dashboard", help="Show fleet dashboard")

    # stats
    p_st = sub.add_parser("stats", help="Device telemetry stats")
    p_st.add_argument("device_id")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    mgr = IoTDeviceManager(db_path=args.db)
    try:
        dispatch = {
            "register": cmd_register,
            "telemetry": cmd_telemetry,
            "alert": cmd_alert,
            "firmware": cmd_firmware,
            "group": cmd_group,
            "dashboard": cmd_dashboard,
            "stats": cmd_stats,
        }
        handler = dispatch.get(args.command)
        if handler:
            handler(args, mgr)
        else:
            parser.print_help()
    finally:
        mgr.close()


if __name__ == "__main__":
    main()
