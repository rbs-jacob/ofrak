# Makefile for building PyInstaller test binaries
# Assumes it's running in a Python virtual environment with PyInstaller installed

.PHONY: all clean simple

all: simple

simple:
	echo "Building PyInstaller test binary with PyInstaller $(shell pyinstaller --version)"
	pyinstaller --onefile --clean simple_hello.py
	mkdir -p ../
	cp dist/simple_hello ../pyinstaller_simple_hello
	echo "Created PyInstaller test binary at ../pyinstaller_simple_hello"

clean:
	echo "Cleaning PyInstaller build artifacts"
	rm -rf build/ dist/ __pycache__/ *.spec
	rm -f ../pyinstaller_simple_hello 
