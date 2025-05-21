import pandas as pd
from typing import Dict, List
import numpy as np

class EPFODataProcessor:
    """Enhanced EPFO data processor with better output formatting and validation"""
    
    def __init__(self, filepath: str, sheet_name: str = "Sheet2"):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.raw_df = None
        self.clean_df = None
        self.mismatches = None
        
        # Define age group sorting order for consistent output
        self.age_group_order = ['<18', '18-21', '22-25', '26-28', '29-35', '>35']
        
        # Column name mapping
        self.column_map = {
            'Number of new EPF subscribers': 'new_subscribers',
            'Number of members exited': 'exited',
            'Number of exited members who rejoined and resubscribed': 'rejoined',
            'Net Payroll': 'net_payroll',
            'Years': 'year',
            'Age': 'age_group'
        }

    def load_and_clean(self) -> pd.DataFrame:
        """Main method to load and clean data with enhanced output"""
        print("="*50)
        print("Starting EPFO Data Processing")
        print("="*50)
        
        try:
            # Load data
            self._load_raw_data()
            
            # Clean and process
            self._clean_data()
            
            # Validate
            self._validate_data()
            
            # Print processing summary
            self._print_processing_summary()
            
            return self.clean_df.copy()
            
        except Exception as e:
            print(f"\n❌ Error in processing: {str(e)}")
            raise

    def _load_raw_data(self):
        """Load data with better error handling"""
        print("\n🔍 Loading raw data...")
        self.raw_df = pd.read_excel(
            self.filepath,
            sheet_name=self.sheet_name,
            engine='openpyxl'
        )
        print(f"✅ Successfully loaded {len(self.raw_df)} records with {len(self.raw_df.columns)} columns")

    def _clean_data(self):
        """Enhanced cleaning with progress tracking"""
        print("\n🧹 Cleaning data...")
        df = self.raw_df.copy()
        
        # Standardize columns
        df = df.rename(columns=self.column_map)
        
        # Clean values
        df['year'] = df['year'].astype(str).str.strip()
        df['age_group'] = (
            df['age_group']
            .str.strip()
            .str.replace('Less than', '<')
            .str.replace('More than', '>')
            .str.replace(' ', '')
        )
        
        # Convert numeric columns
        numeric_cols = ['new_subscribers', 'exited', 'rejoined', 'net_payroll']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Calculate metrics
        df = self._calculate_metrics(df)
        
        self.clean_df = df
        print("✅ Data cleaning completed")

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics with mismatch tracking"""
        # Calculate expected payroll
        df['calculated_payroll'] = df['new_subscribers'] + df['rejoined'] - df['exited']
        
        # Identify mismatches
        self.mismatches = df[df['net_payroll'] != df['calculated_payroll']]
        
        # Calculate rates with zero-division protection
        df['churn_rate'] = np.where(
            (df['new_subscribers'] + df['rejoined']) > 0,
            df['exited'] / (df['new_subscribers'] + df['rejoined']),
            0
        )
        
        df['rejoin_rate'] = np.where(
            df['exited'] > 0,
            df['rejoined'] / df['exited'],
            0
        )
        
        return df.drop(columns=['calculated_payroll'])

    def _validate_data(self):
        """Enhanced validation with clear output"""
        print("\n🔎 Validating data...")
        issues = []
        
        # Check for negative values (except net_payroll)
        numeric_cols = ['new_subscribers', 'exited', 'rejoined']
        for col in numeric_cols:
            if (self.clean_df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for calculation mismatches
        if not self.mismatches.empty:
            issues.append(f"{len(self.mismatches)} net_payroll calculation mismatches")
        
        if issues:
            print("⚠️ Validation issues found:")
            for issue in issues:
                print(f" - {issue}")
        else:
            print("✅ All data validated successfully")

    def _print_processing_summary(self):
        """Beautifully formatted processing summary"""
        print("\n" + "="*50)
        print("📊 Processing Summary")
        print("="*50)
        
        # Years coverage
        years = sorted(self.clean_df['year'].unique())
        print(f"\n📅 Years Covered: {len(years)} years ({min(years)} to {max(years)})")
        
        # Age groups (sorted properly)
        age_groups = sorted(
            self.clean_df['age_group'].unique(),
            key=lambda x: self.age_group_order.index(x)
        )
        print(f"👥 Age Groups: {', '.join(age_groups)}")
        
        # Key metrics
        print("\n🔢 Key Metrics:")
        print(f" - Total records: {len(self.clean_df):,}")
        print(f" - New subscribers: {self.clean_df['new_subscribers'].sum():,}")
        print(f" - Exits: {self.clean_df['exited'].sum():,}")
        print(f" - Rejoins: {self.clean_df['rejoined'].sum():,}")
        
        # Net payroll range
        min_payroll = self.clean_df['net_payroll'].min()
        max_payroll = self.clean_df['net_payroll'].max()
        print(f"\n💰 Net Payroll Range: {min_payroll:,} to {max_payroll:,}")
        
        # Sample data
        print("\n🔍 Sample Data:")
        print(self.clean_df.head().to_markdown(tablefmt="grid", index=False))

    def get_summary_stats(self) -> Dict:
        """Enhanced summary statistics with proper formatting"""
        if self.clean_df is None:
            raise RuntimeError("Data not cleaned yet. Call load_and_clean() first.")
            
        # Sort age groups according to our defined order
        present_groups = [g for g in self.age_group_order if g in self.clean_df['age_group'].unique()]
        
        return {
            'years': sorted(self.clean_df['year'].unique().tolist()),
            'age_groups': present_groups,
            'total_records': len(self.clean_df),
            'total_new_subscribers': int(self.clean_df['new_subscribers'].sum()),
            'total_exits': int(self.clean_df['exited'].sum()),
            'total_rejoins': int(self.clean_df['rejoined'].sum()),
            'net_payroll_range': (
                int(self.clean_df['net_payroll'].min()),
                int(self.clean_df['net_payroll'].max())
            ),
            'calculation_mismatches': len(self.mismatches) if self.mismatches is not None else 0
        }