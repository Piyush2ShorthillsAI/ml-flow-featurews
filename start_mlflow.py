#!/usr/bin/env python3
"""
start_mlflow_cli.py

Start MLflow Tracking Server programmatically via the mlflow CLI entrypoint.

Requirements:
    pip install mlflow python-dotenv

Place a .env file next to this script (example below).
"""

import os
import sys
from dotenv import load_dotenv
from mlflow.server.auth.client import AuthServiceClient
import mlflow.cli

# Load .env from the script directory (or current working dir)
load_dotenv()

# Required values read from environment / .env
secret_key = os.getenv("MLFLOW_FLASK_SERVER_SECRET_KEY")
admin_user = os.getenv("MLFLOW_ADMIN_USER")
admin_pass = os.getenv("MLFLOW_ADMIN_PASSWORD")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./artifacts")
host = os.getenv("MLFLOW_HOST", "0.0.0.0")
port = os.getenv("MLFLOW_PORT", "5000")

# Basic validation
if not secret_key:
    raise SystemExit("ERROR: MLFLOW_FLASK_SERVER_SECRET_KEY is required in .env")

if not admin_user or not admin_pass:
    raise SystemExit("ERROR: MLFLOW_ADMIN_USER and MLFLOW_ADMIN_PASSWORD are required in .env")

# Ensure loaded variables are visible to mlflow CLI invocation
os.environ["MLFLOW_FLASK_SERVER_SECRET_KEY"] = secret_key
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root
print("In mlflow script")
# Optionally export any MLflow-specific env vars here if needed
# os.environ["SOME_OTHER_MLFLOW_VAR"] = "value"

# Create admin user using MLflow auth client (if auth backend is available)
client = AuthServiceClient(tracking_uri)
try:
    client.create_user(admin_user, admin_pass)
    client.assign_role_to_user(admin_user, "admin")
    print(f"[OK] Created admin user: {admin_user}")
except Exception as e:
    # If user already exists, AuthServiceClient will typically raise; ignore
    print(f"[OK] Admin user probably already exists ({admin_user}) — continuing. ({e})")

cli_args = [
    "--backend-store-uri", tracking_uri,
    "--default-artifact-root", artifact_root,
    "--host", host,
    "--port", str(port),
    # you can add more CLI flags here, e.g. "--gunicorn-opts", "workers=2"
]

print("Starting MLflow server via mlflow.cli.server(...)")
print(f"  URL: http://{host}:{port}")
print(f"  backend store: {tracking_uri}")
print(f"  artifact root: {artifact_root}")

# mlflow.cli.server is a Click command function — call it with the args list
    # Some MLflow versions expect a 'args' list passed directly into the function call.
    # This invocation is equivalent to running: `mlflow server <flags...>`
mlflow.cli.server(cli_args)
# Older/newer MLflow might expose different signature; try invoking without args (fallback)
