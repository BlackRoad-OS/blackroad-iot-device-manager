# BlackRoad IoT Device Manager

![CI](https://github.com/BlackRoad-OS/blackroad-iot-device-manager/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-proprietary-red)
![Platform](https://img.shields.io/badge/platform-BlackRoad%20OS-black)

> Comprehensive IoT device lifecycle management — register devices, stream telemetry,
> enforce alert rules, orchestrate firmware OTA updates, and monitor your entire fleet
> from a single SQLite-backed CLI.

## Features

- **Device Registry** — register, group, and track any IoT device type
- **Telemetry Ingestion** — bulk-insert metrics with automatic alert evaluation
- **Alert Rules** — threshold-based rules with configurable severity levels
- **Firmware OTA Queue** — queue and simulate over-the-air firmware updates
- **Fleet Dashboard** — instant overview of online/offline counts and pending jobs
- **Anomaly Detection** — z-score-based outlier detection on sensor streams

## Installation

```bash
git clone https://github.com/BlackRoad-OS/blackroad-iot-device-manager.git
cd blackroad-iot-device-manager
pip install pytest pytest-cov
```

## Usage

### Register a device
```bash
python src/iot_device_manager.py register "temp-sensor-01" --type temperature --ip 192.168.1.10 --firmware 2.0.0
```

### Ingest telemetry
```bash
python src/iot_device_manager.py telemetry <device_id> temperature:72.5:F humidity:45.2:%
```

### Add an alert rule
```bash
python src/iot_device_manager.py alert <device_id> temperature ">" 80.0 --severity critical
```

### Queue a firmware update
```bash
python src/iot_device_manager.py firmware queue <device_id> 3.1.0 sha256:abc123
```

### Process firmware queue
```bash
python src/iot_device_manager.py firmware process --limit 10
```

### View fleet dashboard
```bash
python src/iot_device_manager.py dashboard
```

## Running Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
