#!/usr/bin/env python3
"""
DD-POC Launch Script - Python Version
Simple script to start the Streamlit application with uv
"""

import os
import sys
import subprocess
from pathlib import Path


def check_uv_installation():
    """Check if uv is installed"""
    try:
        result = subprocess.run(['uv', '--version'],
                              capture_output=True,
                              text=True,
                              check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_uv_command(command, cwd=None):
    """Run a uv command and return success status"""
    try:
        result = subprocess.run(['uv'] + command,
                              cwd=cwd,
                              check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ uv command failed: {' '.join(command)}")
        return False


def main():
    """Main launch function"""
    print("🚀 Starting DD-POC Application...")
    print("")

    # Check if uv is installed
    if not check_uv_installation():
        print("❌ Error: uv is not installed")
        print("   Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    # Install dependencies if needed
    print("📦 Ensuring dependencies are installed...")
    if not run_uv_command(['sync']):
        sys.exit(1)

    print("")
    print("🌐 Starting Streamlit server...")
    print("   App will open in your browser automatically")
    print("   Press Ctrl+C to stop the server")
    print("")

    # Enable tokenizers parallelism for better performance
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    # Ensure we're in the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Add current directory to Python path so app module can be found
    python_path = os.environ.get('PYTHONPATH', '')
    if python_path:
        python_path = f"{python_path}:{project_root}"
    else:
        python_path = str(project_root)

    # Set PYTHONPATH for the subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = python_path

    # Start the application with proper PYTHONPATH
    try:
        subprocess.run(['uv', 'run', 'streamlit', 'run', 'app/main.py'],
                      cwd=project_root,
                      env=env,
                      check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
