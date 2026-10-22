"""The monitoring dashboard is the public landing page in services mode."""

from monitor.app import create_app
from src.vercel import deployment_settings

app = create_app(deployment_settings())
