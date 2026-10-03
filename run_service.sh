#!/bin/bash
cd "/Users/mrohithdharshan/Downloads/ddos-cloud-detector" || exit 1
exec ".venv/bin/uvicorn" src.service:app --host 0.0.0.0 --port 8010
