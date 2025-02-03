from src.analyzer import DocumentAnalyzer
from config.config import Config
import argparse

def main():
    parser = argparse.ArgumentParser(description='Document Analysis System')
    parser.add_argument('--input', required=True, help='Path to input document')
    parser.add_argument('--output', default='analysis_results.csv', help='Path to output CSV')
    args = parser.parse_args()

    # Initialize analyzer
    config = vars(Config)
    analyzer = DocumentAnalyzer(config)

    # Analyze document
    try:
        results = analyzer.analyze_document(args.input)
        results.to_csv(args.output, index=False)
        print(f"Analysis complete. Results saved to {args.output}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()