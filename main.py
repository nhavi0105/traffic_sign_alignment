# main.py
# Main entry point for the Traffic Sign Alignment Pipeline

from pipeline.processor import TrafficSignProcessor
import config as config

def main():
    processor = TrafficSignProcessor()
    processor.run_pipeline()

if __name__ == "__main__":
    main()