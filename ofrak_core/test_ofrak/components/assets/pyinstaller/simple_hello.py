#!/usr/bin/env python3
"""
Simple script for testing PyInstaller unpacking.
"""


def main():
    print("Hello from PyInstaller packed script!")

    # Add some example modules to import
    import sys
    import os
    import json

    # Print some information
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")

    # Create some data structures
    data = {"message": "Hello from PyInstaller", "version": "1.0", "test": True}

    print(f"Data: {json.dumps(data, indent=2)}")


if __name__ == "__main__":
    main()
