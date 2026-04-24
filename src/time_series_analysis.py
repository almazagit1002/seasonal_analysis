import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.s3_utils import read_s3_file
from utils.config_reader import ConfigReader
from utils.logger import get_logger

logger = get_logger(__name__, log_to_file=True)

class SeasonalAnalyzer:
    """
    Financial Time Series Analyzer:
    Performs Multiplicative Decomposition, generates trading signals based on 
    trend/noise deviations, and extracts historical seasonal edges.
    """

    def __init__(self, config_file: str = "config"):
        try:
            logger.info("Initializing SeasonalAnalyzer...")
            self.cfg_reader = ConfigReader(config_file)
            
            self.data_cfg = self.cfg_reader.get("data")
            if not self.data_cfg:
                raise ValueError("Missing 'data' section in config")

            self.bucket = self.data_cfg.get("bucket")
            self.key = self.data_cfg.get("bronze_key")
            
            # Setup plots directory
            self.plot_dir = "plots"
            os.makedirs(self.plot_dir, exist_ok=True)
            
            self.df = None
            self.decomposition_result = None
            logger.info(f"Analyzer ready for s3://{self.bucket}/{self.key}")

        except Exception:
            logger.exception("Failed to initialize SeasonalAnalyzer")
            raise

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Downloads and prepares data for statistical analysis."""
        try:
            logger.info(f"Downloading data from S3...")
            self.df = read_s3_file(bucket=self.bucket, key=self.key)

            # Auto-detect Date column
            date_col = next((c for c in self.df.columns if c.lower() in ["date", "timestamp", "time"]), None)
            if not date_col:
                date_col = self.df.select_dtypes(include=['datetime64']).columns[0]
            
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).set_index(date_col)
            
            # Ensure strict daily frequency (fills gaps if any)
            self.df = self.df.asfreq('D').ffill()
            
            logger.info(f"Data prepared: {len(self.df)} rows.")
            return self.df

        except Exception:
            logger.exception("Error during data preparation")
            raise

    def decompose(self, column_name: str = None, model: str = "multiplicative", period: int = 365):
        """
        Decomposes the series and saves a visual report to the /plots folder.
        """
        try:
            if self.df is None: self.load_and_prepare_data()

            if not column_name:
                column_name = self.df.select_dtypes(include=[np.number]).columns[0]

            logger.info(f"Running {model} decomposition on {column_name}...")

            # Multiplicative safety check
            if model == "multiplicative" and (self.df[column_name] <= 0).any():
                logger.warning("Zero/Negative values detected. Using additive model.")
                model = "additive"

            self.decomposition_result = seasonal_decompose(
                self.df[column_name], 
                model=model, 
                period=period
            )

            # Save Plot
            fig = self.decomposition_result.plot()
            fig.set_size_inches(14, 10)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.plot_dir, f"decomp_{column_name}_{timestamp}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Decomposition complete. Plot: {save_path}")
            return self.decomposition_result

        except Exception:
            logger.exception("Decomposition failed")
            raise

 
    def get_seasonal_calendar(self) -> pd.DataFrame:
        """
        Calculates the average historical performance 'edge' for each month.
        """
        try:
            if self.decomposition_result is None:
                raise ValueError("Must run decompose() before calendar.")

            logger.info("Extracting monthly seasonal profile...")
            
            seasonal_df = pd.DataFrame({'factor': self.decomposition_result.seasonal}, index=self.df.index)
            seasonal_df['month'] = seasonal_df.index.month
            
            # Group by month and calculate the average edge
            monthly_profile = seasonal_df.groupby('month')['factor'].mean().reset_index()
            
            month_map = {
                1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
            }
            monthly_profile['month_name'] = monthly_profile['month'].map(month_map)
            monthly_profile['historical_edge_pct'] = (monthly_profile['factor'] - 1) * 100
            
            # Sort by performance
            monthly_profile = monthly_profile.sort_values('historical_edge_pct', ascending=False)
            
            return monthly_profile[['month_name', 'historical_edge_pct']]

        except Exception:
            logger.exception("Failed to generate seasonal calendar")
            raise

    def get_current_prediction(self, monthly_profile: pd.DataFrame):
        """Prints a summary prediction for the current month."""
        month_name = datetime.now().strftime("%b")
        row = monthly_profile[monthly_profile['month_name'] == month_name].iloc[0]
        edge = row['historical_edge_pct']
        
        print("\n" + "="*40)
        print(f"SEASONAL INSIGHT: {month_name.upper()}")
        print("-" * 40)
        if edge > 2:
            print(f"STATUS: BULLISH")
        elif edge < -2:
            print(f"STATUS: CAUTIOUS")
        else:
            print(f"STATUS: NEUTRAL")
        print(f"Historical Advantage: {edge:+.2f}%")
        print("="*40 + "\n")

    def get_halving_analysis(self) -> dict:
        """
        Determines the current position in the 4-year Bitcoin Halving Cycle.
        """
        try:
            # Historical Halving Dates
            halvings = [
                datetime(2012, 11, 28),
                datetime(2016, 7, 9),
                datetime(2020, 5, 11),
                datetime(2024, 4, 20),
                datetime(2028, 3, 1)  # Estimated next
            ]
            
            now = datetime.now()
            # Find the last halving
            last_halving = max([h for h in halvings if h <= now])
            days_since = (now - last_halving).days
            
            # Historical cycle logic (approximate phases)
            if days_since <= 450:
                phase = "BULL MARKET (Post-Halving Surge)"
                desc = "Historically the period of maximum price discovery."
            elif 450 < days_since <= 800:
                phase = "BEAR MARKET / CORRECTION"
                desc = "Historically the period of 'cooling off' from the peak."
            else:
                phase = "ACCUMULATION (Pre-Halving)"
                desc = "Price tends to stabilize as the next halving approaches."

            return {
                "last_halving": last_halving.strftime("%Y-%m-%d"),
                "days_since": days_since,
                "cycle_phase": phase,
                "phase_description": desc
            }
        except Exception:
            logger.exception("Halving analysis failed")
            return {}


if __name__ == "__main__":
    try:
        analyzer = SeasonalAnalyzer()
        
        # 1. Run Analysis
        analyzer.decompose(model="multiplicative")
        
        # 2. Get Numerical Tables
     
        calendar = analyzer.get_seasonal_calendar()
        
    
        
        print("\nANNUAL SEASONAL CALENDAR (Ranked):")
        print(calendar)
        
        # 4. Current Month Logic
        analyzer.get_current_prediction(calendar)

        # 5. Halving Logic
        halving_data = analyzer.get_halving_analysis()
        print("\n" + "="*40)
        print(f"BITCOIN HALVING CYCLE DATA")
        print("-" * 40)
        print(f"Last Halving: {halving_data['last_halving']}")
        print(f"Days Since:   {halving_data['days_since']}")
        print(f"CURRENT PHASE: {halving_data['cycle_phase']}")
        print(f"Context:      {halving_data['phase_description']}")
        print("="*40 + "\n")

    except Exception:
        logger.error("Analyzer script failed.")