#!/usr/bin/env python3
"""
Environment setup script for STV transfer analysis.

This script helps users set up their environment for running STV transfer
analysis, including JAX installation detection and basic validation.
"""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or later is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_jax_installation():
    """Check if JAX is properly installed."""
    try:
        import jax
        import jax.numpy as jnp

        # Test basic functionality
        x = jnp.array([1, 2, 3])
        y = jnp.sum(x)

        print(f"✅ JAX {jax.__version__} installed and working")
        print(f"   Backend: {jax.lib.xla_bridge.get_backend().platform}")
        return True

    except ImportError:
        print("❌ JAX not found")
        return False
    except Exception as e:
        print(f"❌ JAX installation issue: {e}")
        return False


def check_numpyro():
    """Check if NumPyro is available."""
    try:
        import numpyro
        print(f"✅ NumPyro {numpyro.__version__}")
        return True
    except ImportError:
        print("❌ NumPyro not found")
        return False


def suggest_jax_installation():
    """Suggest appropriate JAX installation command."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    print("\n📦 JAX Installation Suggestions:")
    print("=" * 40)

    if system == "windows":
        print("🪟 Windows detected")
        print("   Recommendation: Use Windows Subsystem for Linux (WSL)")
        print("   JAX has limited native Windows support")
        print("\n   If you must use Windows:")
        print("   pip install 'jax[cpu]'")

    elif system == "darwin":  # macOS
        print("🍎 macOS detected")
        if "arm" in machine or "m1" in machine or "m2" in machine:
            print("   Apple Silicon detected")
            print("   pip install 'jax[metal]'")
        else:
            print("   Intel Mac detected")
            print("   pip install 'jax[cpu]'")

    elif system == "linux":
        print("🐧 Linux detected")
        print("   For CPU: pip install 'jax[cpu]'")
        print("   For NVIDIA GPU: pip install 'jax[cuda12]'")
        print("   (Use jax[cuda11] for older CUDA versions)")

    else:
        print(f"❓ Unknown system: {system}")
        print("   Try: pip install 'jax[cpu]'")


def check_package_structure():
    """Check if package structure is correct."""
    root = Path(__file__).parent.parent

    required_paths = [
        "src/stv_transfers/__init__.py",
        "tests/test_smoke.py",
        "pyproject.toml",
        "README.md"
    ]

    all_good = True
    for path_str in required_paths:
        path = root / path_str
        if path.exists():
            print(f"✅ {path_str}")
        else:
            print(f"❌ Missing: {path_str}")
            all_good = False

    return all_good


def run_smoke_tests():
    """Run basic smoke tests."""
    root = Path(__file__).parent.parent

    print("\n🧪 Running smoke tests...")
    print("=" * 30)

    try:
        # Add src to Python path
        sys.path.insert(0, str(root / "src"))

        # Try basic imports
        import stv_transfers
        print(f"✅ Package import successful (v{stv_transfers.__version__})")

        from stv_transfers import data_structures, scraper, simulator, diagnostics
        print("✅ Core modules imported successfully")

        try:
            from stv_transfers import model
            print("✅ Model module imported (NumPyro available)")
        except ImportError as e:
            if "NumPyro" in str(e) or "JAX" in str(e):
                print("⚠️  Model module requires NumPyro/JAX")
            else:
                print(f"❌ Model import error: {e}")

        print("✅ All basic tests passed")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup check function."""
    print("🔬 STV Transfer Analysis Environment Check")
    print("=" * 45)

    all_checks = []

    # System checks
    print("\n📋 System Requirements:")
    all_checks.append(check_python_version())
    all_checks.append(check_jax_installation())
    all_checks.append(check_numpyro())

    # Package structure
    print("\n📁 Package Structure:")
    all_checks.append(check_package_structure())

    # JAX installation help
    if not check_jax_installation():
        suggest_jax_installation()

    # Smoke tests
    all_checks.append(run_smoke_tests())

    # Summary
    print("\n" + "=" * 45)
    if all(all_checks):
        print("🎉 Environment setup complete! You're ready to go.")
        print("\n📚 Next steps:")
        print("   1. Install JAX if not already done (see suggestions above)")
        print("   2. pip install -e '.[dev]' to install in development mode")
        print("   3. export PYTHONPATH=src:$PYTHONPATH")
        print("   4. Check out the README.md for usage examples")
    else:
        print("⚠️  Some issues found. Please address them before proceeding.")
        print("   See suggestions above for next steps.")

    return all(all_checks)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)