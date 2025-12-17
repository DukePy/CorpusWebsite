#!/bin/bash

# Startup script for Corpus Website

echo "========================================="
echo "  Digital Gateway to Asian Literature"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "Starting Flask application..."
echo ""
echo "The application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask application
python corpusFunctions.py

