"""
Main entry point for AACI worker.
"""
import sys
import logging
from aaci.workers import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    main()
