import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys
import re

class PUSCHMLAnalysis:
    def __init__(self, file_path):
        """Initialize the ML analysis pipeline."""
        self.file_path = file_path
        self.raw_data = None
        self.df = None
        self.X = None
        self.y = None
        self.models = {}

    def extract_value(self, parts, key):
        """
        Safely extract value for a given key from parts list.
        """
        try:
            for i, part in enumerate(parts):
                if key in part:
                    return parts[i + 1]
        except IndexError:
            return None
        return None

    def parse_pusch_log(self):
        """Parse PUSCH log file and return list of dictionaries."""
        results = []
        current_entry = {}

        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    # Extract timestamp
                    timestamp_match = re.search(r'\[(.*?)\]', line)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)

                    # Split the line by colons
                    parts = [p.strip() for p in line.strip().split(':')]

                    # Process line type 1 (with CELLNUM, SFN, etc.)
                    if 'PUSCHIND_CELLNUM' in line:
                        current_entry = {
                            'timestamp': timestamp,
                            'CELLNUM': self.extract_value(parts, 'PUSCHIND_CELLNUM'),
                            'SFN': self.extract_value(parts, 'PUSCHIND_SFN'),
                            'SLOTNUM': self.extract_value(parts, 'PUSCHIND_SLOTNUM'),
                            'UEID': self.extract_value(parts, 'PUSCHIND_UEID'),
                            'RNTI': self.extract_value(parts, 'PUSCHIND_RNTI'),
                            'CRCRESULT': self.extract_value(parts, 'PUSCHIND_CRCRESULT'),
                            'INSTTAEST': self.extract_value(parts, 'PUSCHIND_INSTTAEST'),
                            'INSTSNR': self.extract_value(parts, 'PUSCHIND_INSTSNR'),
                            'POSTSNR': self.extract_value(parts, 'PUSCHIND_POSTSNR')
                        }

                    # Process line type 2 (with AVGTAEST, etc.)
                    elif 'PUSCHIND_AVGTAEST' in line:
                        if current_entry:  # Only if we have a valid current entry
                            additional_data = {
                                'AVGTAEST': self.extract_value(parts, 'PUSCHIND_AVGTAEST'),
                                'AVGSNR': self.extract_value(parts, 'PUSCHIND_AVGSNR'),
                                'FOE': self.extract_value(parts, 'PUSCHIND_FOE'),
                                'FOC': self.extract_value(parts, 'PUSCHIND_FOC'),
                                'DOPP': self.extract_value(parts, 'PUSCHIND_DOPP')
                            }

                            # Only add entries if we have valid data
                            if all(v is not None for v in additional_data.values()):
                                current_entry.update(additional_data)
                                results.append(current_entry.copy())
                            current_entry = {}

        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)

        return results

    def load_and_preprocess_data(self):
        """Load and preprocess the PUSCH log data."""
        print("Parsing log file...")
        parsed_data = self.parse_pusch_log()

        if not parsed_data:
            print("No valid data found in the log file.")
            sys.exit(1)

        print(f"Found {len(parsed_data)} valid entries")

        # Convert to DataFrame
        self.raw_data = pd.DataFrame(parsed_data)

        print("\nInitial columns:", self.raw_data.columns.tolist())

        # Convert relevant columns to numeric
        numeric_columns = ['AVGTAEST', 'AVGSNR', 'INSTSNR', 'POSTSNR',
                         'FOE', 'FOC', 'DOPP', 'CRCRESULT']

        for col in numeric_columns:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            else:
                print(f"Warning: Column {col} not found in data")

        # Extract time features
        try:
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            self.raw_data['hour'] = self.raw_data['timestamp'].dt.hour
            self.raw_data['minute'] = self.raw_data['timestamp'].dt.minute
        except Exception as e:
            print(f"Warning: Could not process timestamp: {str(e)}")
            self.raw_data['hour'] = 0
            self.raw_data['minute'] = 0

        # Handle missing values
        self.df = self.raw_data.copy()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        print("\nData Preprocessing Summary:")
        print("-" * 80)
        print(f"Total samples: {len(self.df)}")
        print("\nColumns in processed data:", self.df.columns.tolist())
        print("\nFeature statistics:")
        print(self.df.describe())

        return self.df

    def analyze_features(self):
        """Perform feature analysis and visualization."""
        if self.df is None or len(self.df) == 0:
            print("No data available for analysis")
            return

        plt.figure(figsize=(15, 10))

        # Correlation matrix
        plt.subplot(2, 2, 1)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')

        # AVGTAEST vs AVGSNR scatter plot
        if 'AVGSNR' in self.df.columns and 'AVGTAEST' in self.df.columns:
            plt.subplot(2, 2, 2)
            plt.scatter(self.df['AVGSNR'], self.df['AVGTAEST'], alpha=0.5)
            plt.xlabel('AVGSNR')
            plt.ylabel('AVGTAEST')
            plt.title('AVGTAEST vs AVGSNR')

        # CRC Result distribution
        if 'CRCRESULT' in self.df.columns:
            plt.subplot(2, 2, 3)
            self.df['CRCRESULT'].value_counts().plot(kind='bar')
            plt.title('CRC Result Distribution')
            plt.xlabel('CRC Result')
            plt.ylabel('Count')

        # SNR distributions
        snr_cols = [col for col in ['AVGSNR', 'INSTSNR', 'POSTSNR'] if col in self.df.columns]
        if snr_cols:
            plt.subplot(2, 2, 4)
            sns.kdeplot(data=self.df[snr_cols])
            plt.title('SNR Distributions')
            plt.xlabel('SNR Value')

        plt.tight_layout()
        plt.show()

    # [Rest of the class methods remain the same...]

def main():
    parser = argparse.ArgumentParser(description='PUSCH Data ML Analysis')
    parser.add_argument('filename', help='Path to the log file')
    parser.add_argument('--problem-type', choices=['regression', 'classification'],
                       default='regression', help='Type of ML problem to solve')
    args = parser.parse_args()

    # Initialize and run analysis
    analysis = PUSCHMLAnalysis(args.filename)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    analysis.load_and_preprocess_data()

    # Analyze features
    print("\nAnalyzing features...")
    analysis.analyze_features()

    # Prepare features and train models
    print("\nPreparing features and training models...")
    analysis.prepare_features(problem_type=args.problem_type)
    analysis.train_models(problem_type=args.problem_type)

if __name__ == "__main__":
    main()
