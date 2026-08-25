"""Investment strategy agent using A2A protocol."""

from dotenv import find_dotenv, load_dotenv

# Load .env from current working directory or search up from package hierarchy
if not load_dotenv(find_dotenv(usecwd=True)):
    load_dotenv(find_dotenv(usecwd=False))
