#!/usr/bin/env python3
"""
E2E Test Runner Script

Script to run end-to-end tests for the AI Due Diligence application.
Provides options for different test suites and configurations.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description="", timeout=None):
    """Run a command with error handling"""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root
        )
        print("✅ Success")
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, e.stdout, e.stderr
    except subprocess.TimeoutExpired as e:
        print(f"⏰ Timeout after {timeout} seconds")
        return False, "", str(e)


def check_prerequisites():
    """Check that all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check if uv is available
    success, _, _ = run_command(["uv", "--version"], "Checking uv")
    if not success:
        print("❌ uv is not available. Please install uv first.")
        return False
    
    # Check if Playwright browsers are installed
    success, _, _ = run_command(["uv", "run", "playwright", "install", "--dry-run"], "Checking Playwright browsers")
    if not success:
        print("⚠️  Playwright browsers may need to be installed")
        print("Run: uv run playwright install chromium")
    
    # Check if main app file exists
    app_file = project_root / "app" / "main.py"
    if not app_file.exists():
        print(f"❌ Main app file not found: {app_file}")
        return False
    
    print("✅ Prerequisites check completed")
    return True


def run_smoke_tests():
    """Run smoke tests (basic functionality)"""
    cmd = [
        "uv", "run", "pytest", 
        "-c", "pytest-e2e.ini",
        "tests/e2e/test_app_startup.py",
        "-m", "not slow",
        "--maxfail=3"
    ]
    
    return run_command(cmd, "Running smoke tests", timeout=300)


def run_full_tests():
    """Run all E2E tests"""
    cmd = [
        "uv", "run", "pytest", 
        "-c", "pytest-e2e.ini",
        "tests/e2e/",
        "--maxfail=5"
    ]
    
    return run_command(cmd, "Running full E2E test suite", timeout=1200)


def run_performance_tests():
    """Run performance tests"""
    cmd = [
        "uv", "run", "pytest", 
        "-c", "pytest-e2e.ini",
        "tests/e2e/test_performance.py",
        "-m", "not slow"
    ]
    
    return run_command(cmd, "Running performance tests", timeout=600)


def run_ai_tests():
    """Run AI analysis tests"""
    cmd = [
        "uv", "run", "pytest", 
        "-c", "pytest-e2e.ini",
        "tests/e2e/test_ai_analysis.py",
        "-m", "not slow"
    ]
    
    return run_command(cmd, "Running AI analysis tests", timeout=600)


def run_custom_tests(test_path, markers=None):
    """Run custom test selection"""
    cmd = [
        "uv", "run", "pytest", 
        "-c", "pytest-e2e.ini",
        test_path
    ]
    
    if markers:
        cmd.extend(["-m", markers])
    
    return run_command(cmd, f"Running custom tests: {test_path}", timeout=900)


def install_browsers():
    """Install Playwright browsers"""
    cmd = ["uv", "run", "playwright", "install", "chromium"]
    return run_command(cmd, "Installing Playwright browsers", timeout=300)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run E2E tests for AI Due Diligence app")
    parser.add_argument(
        "--suite", 
        choices=["smoke", "full", "performance", "ai", "custom"],
        default="smoke",
        help="Test suite to run (default: smoke)"
    )
    parser.add_argument(
        "--test-path",
        help="Specific test path (for custom suite)"
    )
    parser.add_argument(
        "--markers",
        help="Pytest markers to filter tests (e.g., 'not slow')"
    )
    parser.add_argument(
        "--install-browsers",
        action="store_true",
        help="Install Playwright browsers before running tests"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run tests in headless mode (default: True)"
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run tests in headed mode (for debugging)"
    )
    
    args = parser.parse_args()
    
    print("🚀 AI Due Diligence E2E Test Runner")
    print("=" * 50)
    
    # Set environment variables
    if args.headed:
        os.environ["PLAYWRIGHT_HEADLESS"] = "false"
    else:
        os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            sys.exit(1)
    
    # Install browsers if requested
    if args.install_browsers:
        success, _, _ = install_browsers()
        if not success:
            print("❌ Failed to install browsers")
            sys.exit(1)
    
    # Run selected test suite
    success = False
    
    if args.suite == "smoke":
        success, _, _ = run_smoke_tests()
    elif args.suite == "full":
        success, _, _ = run_full_tests()
    elif args.suite == "performance":
        success, _, _ = run_performance_tests()
    elif args.suite == "ai":
        success, _, _ = run_ai_tests()
    elif args.suite == "custom":
        if not args.test_path:
            print("❌ --test-path is required for custom suite")
            sys.exit(1)
        success, _, _ = run_custom_tests(args.test_path, args.markers)
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✅ E2E tests completed successfully!")
        print("\n💡 Tips:")
        print("  - Run with --headed to see the browser in action")
        print("  - Use --suite=full for comprehensive testing")
        print("  - Use --markers='not slow' to skip long-running tests")
    else:
        print("❌ E2E tests failed!")
        print("\n🔧 Troubleshooting:")
        print("  - Make sure the Streamlit app can start properly")
        print("  - Check that all dependencies are installed")
        print("  - Try running with --install-browsers first")
        print("  - Run individual tests to isolate issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
