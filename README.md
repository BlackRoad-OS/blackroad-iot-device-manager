<!-- BlackRoad SEO Enhanced -->

# ulackroad iot device manager

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad OS](https://img.shields.io/badge/Org-BlackRoad-OS-2979ff?style=for-the-badge)](https://github.com/BlackRoad-OS)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad iot device manager** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


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
