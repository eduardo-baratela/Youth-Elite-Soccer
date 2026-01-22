import pandas as pd
from datetime import datetime

def parse_gender(val):
    """Parse gender value to binary (1=Weiblich/Female, 0=Männlich/Male)"""
    # numeric flags (already 0/1)
    if val in (0, 1):
        return val
 
    s = str(val).strip().lower()
    # map textual genders: look for 'w' (weiblich) or 'm' (maennlich/männlich)
    if 'w' in s:
        return 1
    if 'm' in s:
        return 0
    
    return None


def fill_gender(df, target='Geschlecht', sources=None):
    """Fill missing gender values using source columns"""
    if sources is None:
        sources = ['T1subj_Geschlecht', 'T2subj_Geschlecht', 'T3subj_Geschlecht', 
                   'T25_Geschlecht', 'T27_Geschlecht', 'T29_Geschlecht']
    
    if target in df.columns:
        df[target] = df[target].apply(parse_gender)
        for src in sources:
            if src in df.columns:
                df[target] = df[target].fillna(df[src].apply(parse_gender))
    
    n_missing = df[target].isna().sum()
    print(f"'{target}' missing after filling: {n_missing}")
    return df


def parse_birth_date(val, format_type='ddmmmyy'):
    """
    Parse birth date from various formats
    format_type: 'ddmmmyy' for 'dd/mmm/yy' or 'ddmmyy' for 'dd/mm/yy'
    Returns datetime object or None
    """
    if pd.isna(val):
        return None
    
    # If already datetime, return it
    if isinstance(val, (datetime, pd.Timestamp)):
        return val
    
    s = str(val).strip()
    if not s or s.lower() == 'nan':
        return None
    
    try:
        if format_type == 'ddmmmyy':
            # Format: dd/mmm/yy (e.g., 15/Jan/95)
            return pd.to_datetime(s, format='%d/%b/%y', errors='coerce')
        elif format_type == 'ddmmyy':
            # Format: dd/mm/yy (e.g., 15/01/95)
            return pd.to_datetime(s, format='%d/%m/%y', errors='coerce')
        elif format_type == 'ddmmyyyy':
            # Format: dd.mm.yyyy (e.g., 15.01.1995)
            return pd.to_datetime(s, format='%d.%m.%Y', errors='coerce')
    except:
        return None
    
    return None


def fill_birth_dates(df, target='Geburtstag', year_target='Geburtsjahre'):
    """Fill missing birth dates and birth years"""
    # Sources with their format types
    sources_format = [
        ('T1subj_Geburtstag', 'ddmmmyy'),
        ('T2subj_Geburtstag', 'ddmmmyy'),
        ('T3subj_Geburtstag', 'ddmmyyyy'),
        ('T25_Geburtstag', 'ddmmyy'),
        ('T27_Geburtstag', 'ddmmyy'),
        ('T29_Geburtstag', 'ddmmyy')
    ]
    
    # Parse target column first
    if target in df.columns:
        df[target] = df[target].apply(lambda x: parse_birth_date(x, 'ddmmmyy'))
        
        # Fill from source columns
        for src, fmt in sources_format:
            if src in df.columns:
                mask = df[target].isna()
                if mask.any():
                    df.loc[mask, target] = df.loc[mask, src].apply(lambda x: parse_birth_date(x, fmt))
    
    # Fill birth year from birth date
    if year_target in df.columns and target in df.columns:
        # First parse existing year values
        df[year_target] = pd.to_numeric(df[year_target], errors='coerce')
        
        # Fill missing years from birth dates
        mask = df[year_target].isna() & df[target].notna()
        df.loc[mask, year_target] = df.loc[mask, target].dt.year
    
    # Report missing values
    n_missing_date = df[target].isna().sum() if target in df.columns else 0
    n_missing_year = df[year_target].isna().sum() if year_target in df.columns else 0
    print(f"'{target}' missing after filling: {n_missing_date}")
    print(f"'{year_target}' missing after filling: {n_missing_year}")
    
    return df


def calculate_relative_age(birth_date):
    """
    Calculate relative age as day of birth within the year.
    Returns the day number (1-365/366) within the birth year.
    
    Args:
        birth_date: datetime object or pd.Timestamp
    
    Returns:
        int: Day of year (1-365 or 1-366 for leap years), or None if invalid
    """
    
    try:
        # Convert to datetime if needed
        if isinstance(birth_date, str):
            birth_date = pd.to_datetime(birth_date)
        
        # Get day of year (1-365 or 1-366)
        return birth_date.timetuple().tm_yday
    except:
        return None


def fill_koordinatorengebiet(df, target='Koordinatorengebiet', sources=None):
    """Fill missing Koordinatorengebiet values from T25, T27, T29 columns"""
    if sources is None:
        sources = ['T25_Koordinatorengebiet', 'T27_Koordinatorengebiet', 'T29_Koordinatorengebiet']
    
    if target in df.columns:
        for src in sources:
            if src in df.columns:
                mask = df[target].isna()
                if mask.any():
                    df.loc[mask, target] = df.loc[mask, src]
    
    n_missing = df[target].isna().sum()
    print(f"'{target}' missing after filling: {n_missing}")
    return df


def fill_stuetzpunktname(df, target='Stützpunktname', sources=None):
    """Fill missing Stützpunktname values from T1subj, T2subj, T3subj, T25, T27, T29 columns"""
    if sources is None:
        sources = ['T1subj_Stützpunktname', 'T2subj_Stützpunktname', 'T3subj_Stützpunktname', 
                   'T25_Stützpunktname', 'T27_Stützpunktname', 'T29_Stützpunktname']
    
    if target in df.columns:
        for src in sources:
            if src in df.columns:
                mask = df[target].isna()
                if mask.any():
                    df.loc[mask, target] = df.loc[mask, src]
    
    n_missing = df[target].isna().sum()
    print(f"'{target}' missing after filling: {n_missing}")
    return df


## Split
    
# Create mask for players in this age group across all test periods
def create_ak_subset(clean_data, ak_level, cols_base, filter_feldspieler=True, gender='male'):
    """
    Create a subset dataframe for a specific age group (Altersklasse).
    
    Parameters:
    -----------
    clean_data : pd.DataFrame
        The cleaned dataframe with renamed columns
    ak_level : str
        Age group level (e.g., 'U12', 'U13', 'U14', 'U15')
    cols_base : list
        Base columns to include in the subset
    filter_feldspieler : bool, optional
        If True, only include players where Spielertyp == 'Feldspieler' for the matching time period
    gender : str
        Gender of the subset to include at the file's name
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe for the specified age group
    """
    # Define feature columns for this age group
    cols_features = [
        f'{ak_level}_FR_SL10',f'{ak_level}_FR_SL20',f'{ak_level}_FR_GW', f'{ak_level}_FR_DR', f'{ak_level}_FR_BK', 
        f'{ak_level}_FR_BJ', f'{ak_level}_FR_SC', f'{ak_level}_FR_Grösse', f'{ak_level}_FR_Gewicht'
    ]
    
    # Create mask for players in this age group across all test periods
    if filter_feldspieler:
        mask = (
            (clean_data['T1subj_AK'].astype(str).str.strip() == ak_level) & 
            (clean_data['T1subj_Spielertyp'].astype(str).str.strip() == 'Feldspieler') &
            (clean_data['T25_AK'].astype(str).str.strip() == ak_level)
        ) | (
            (clean_data['T2subj_AK'].astype(str).str.strip() == ak_level) & 
            (clean_data['T2subj_Spielertyp'].astype(str).str.strip() == 'Feldspieler') &
            (clean_data['T27_AK'].astype(str).str.strip() == ak_level)
        ) | (
            (clean_data['T3subj_AK'].astype(str).str.strip() == ak_level) & 
            (clean_data['T3subj_Spielertyp'].astype(str).str.strip() == 'Feldspieler') &
            (clean_data['T29_AK'].astype(str).str.strip() == ak_level)
        )
    else:
        mask = (
            (clean_data['T1subj_AK'].astype(str).str.strip() == ak_level) &
            (clean_data['T25_AK'].astype(str).str.strip() == ak_level)
        ) | (
            (clean_data['T2subj_AK'].astype(str).str.strip() == ak_level) &
            clean_data['T27_AK'].astype(str).str.strip() == ak_level
        ) | (
            (clean_data['T3subj_AK'].astype(str).str.strip() == ak_level) &
            (clean_data['T29_AK'].astype(str).str.strip() == ak_level)
        )
    
    # Select columns
    ak_cols = cols_base + cols_features
    ak_df = clean_data.loc[mask, ak_cols].copy()
    
    # Save to CSV
    ak_df.to_csv(f'../data/{ak_level.lower()}_data_{gender}.csv', index=False)
    
    print(f"{ak_level} subset created: {len(ak_df)} rows")
    return ak_df


## Refinement

def calculate_subjective_scores(df, time_period):
    """
    Calculate composite subjective scores from individual assessment items for a specific time period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the subjective assessment columns
    time_period : str
        Time period prefix (e.g., 'T1subj', 'T2subj', 'T3subj')
    
    Returns:
    --------
    dict
        Dictionary with keys 'SKSC_TEC', 'SKSC_KON', 'SKSC_TAK', 'SKSC_PSY' 
        containing the averaged scores
    """
    # Define the items for each composite score
    score_items = {
        'TEC': ['Technik_Dom_Fuss', 'Technik_Nicht_Dom_Fuss', 'Kopfballtechnik'],
        'KON': ['Kond_Fähigkeiten'],
        'TAK': ['Taktik_offensiv_vor', 'Taktik_offensiv_während', 'Taktik_offensiv_nach',
                'Taktik_defensiv_vor', 'Taktik_defensiv_während', 'Taktik_defensiv_nach',
                'Spielintelligenz'],
        'PSY': ['Psy_Motivation', 'Psy_Volition', 'Psy_Sozial']
    }
    
    results = {}
    
    for score_type, items in score_items.items():
        # Collect all relevant columns across time periods
        cols = [f'{time_period}_{item}' for item in items if f'{time_period}_{item}' in df.columns]
        
        # Calculate mean across all columns (ignoring NaN)
        results[f'SKSC_{score_type}'] = df[cols].mean(axis=1)
    
    return results

def refine_ak_dataset(ak_df, ak_level, gender="male"):
    """
    Refine age group dataset with standardized column names.
    
    Parameters:
    -----------
    ak_df : pd.DataFrame
        The age group dataframe (U12, U13, U14, U15)
    ak_level : str
        Age level (e.g., 'U12', 'U13', 'U14', 'U15')
    gender : str
        Gender of the subset to include at the file's name
    
    Returns:
    --------
    pd.DataFrame
        Refined dataframe with standardized columns
    """
    # Reset index to avoid duplicate index issues
    ak_df = ak_df.reset_index(drop=True)
    
    refined_df = pd.DataFrame()
    
    # Determine which time period corresponds to this ak_level for each player
    time_period_cols = ['T1subj_AK', 'T2subj_AK', 'T3subj_AK']
    
    def get_time_period_for_row(row):
        """Determine the time period where this player has the target ak_level"""
        for col in time_period_cols:
            if col in ak_df.columns and pd.notna(row[col]) and str(row[col]).strip() == ak_level:
                return col.replace('_AK', '')
        return None
    
    # Get time period for each player
    time_periods = ak_df.apply(get_time_period_for_row, axis=1)
        
    # Add basic columns
    refined_df['TalentID'] = ak_df['TalentID'].values
    refined_df['relative_age'] = ak_df['relative_age'].values
    refined_df['LZ'] = ak_df['LZ'].values
    refined_df['AK'] = ak_level
    
    # Map physical test columns using the ak_level (U12, U13, U14, U15)
    # These come from FR (Fitness Ranking) columns
    col_mapping = {
        f'{ak_level}_FR_Grösse': 'height',
        f'{ak_level}_FR_Gewicht': 'weight',
        f'{ak_level}_FR_SL20': 'SL20',
        f'{ak_level}_FR_GW': 'GW',
        f'{ak_level}_FR_DR': 'DR',
        f'{ak_level}_FR_BK': 'BK',
        f'{ak_level}_FR_BJ': 'BJ',
        f'{ak_level}_FR_SC': 'SC',
    }
    
    for source_col, target_col in col_mapping.items():
        if source_col in ak_df.columns:
            refined_df[target_col] = ak_df[source_col].values

    
    # Calculate SKSC columns (subjective scoring criteria) for each player individually
    def calc_sksc_for_row(idx):
        row = ak_df.iloc[idx]
        time_period = time_periods.iloc[idx]
        if time_period is None:
            return {'SKSC_TEC': None, 'SKSC_KON': None, 'SKSC_TAK': None, 'SKSC_PSY': None}
        
        # Create a single-row dataframe for calculation
        row_df = pd.DataFrame([row])
        scores = calculate_subjective_scores(row_df, time_period)
        return {key: scores[key].iloc[0] for key in scores}
    
    # Calculate scores for each row
    sksc_results = [calc_sksc_for_row(i) for i in range(len(ak_df))]
    
    refined_df['SKSC_TEC'] = [r['SKSC_TEC'] for r in sksc_results]  # Technik (Technical)
    refined_df['SKSC_KON'] = [r['SKSC_KON'] for r in sksc_results]  # Kondition (Conditioning)
    refined_df['SKSC_TAK'] = [r['SKSC_TAK'] for r in sksc_results]  # Taktik (Tactical)
    refined_df['SKSC_PSY'] = [r['SKSC_PSY'] for r in sksc_results]  # Psychologisch (Psychological)
    
    # Reorder columns to match desired order
    desired_cols = ['TalentID', 'AK', 'relative_age', 'height', 'weight', 'SL20', 'GW', 'DR', 'BK', 'BJ',
                    'SKSC_TAK', 'SKSC_TEC', 'SKSC_KON', 'SKSC_PSY', 'LZ']
    # Only include columns that exist
    desired_cols = [col for col in desired_cols if col in refined_df.columns]
    refined_df = refined_df[desired_cols]

    # Fill missing values
    refined_df.fillna(refined_df.mean(numeric_only=True) ,inplace = True)

    # Save to CSV
    refined_df.to_csv(f'../data/refined_{ak_level.lower()}_data_{gender}.csv', index=False)
    
    return refined_df


def merge_refined_ak_datasets(ak_levels=['U12', 'U13', 'U14', 'U15'], data_dir='../data', gender='male'):
    """
    Merge refined AK datasets, keeping only first (earliest) appearance per TalentID.
    
    Parameters:
    -----------
    ak_levels : list
        AK levels in priority order (first = highest priority)
    data_dir : str
        Directory containing refined CSV files
    gender : str
        Gender of the subset to include at the file's name
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with unique TalentIDs
    """
    dfs = []
    for ak in ak_levels:
        csv_path = f'{data_dir}/refined_{ak.lower()}_data_{gender}.csv'
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {ak}")
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found, skipping")
    
    if not dfs:
        raise ValueError("No refined datasets found")
    
    # Concatenate all datasets
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before deduplication: {len(combined)}")
    
    # Create priority mapping (lower number = higher priority)
    ak_priority = {ak: i for i, ak in enumerate(ak_levels)}
    combined['_priority'] = combined['AK'].map(ak_priority)
    
    # Sort by TalentID and priority, then keep first occurrence
    combined_sorted = combined.sort_values(['TalentID', '_priority'])
    merged = combined_sorted.drop_duplicates(subset='TalentID', keep='first')
    merged = merged.drop(columns=['_priority'])
    
    print(f"Total rows after deduplication: {len(merged)}")
    print(f"\nDistribution by AK:")
    print(merged['AK'].value_counts().sort_index())
    
    # Save merged dataset
    merged.to_csv(f'{data_dir}/merged_{gender}.csv', index=False)
    print(f"\nSaved to {data_dir}/merged_{gender}.csv")
    
    return merged
