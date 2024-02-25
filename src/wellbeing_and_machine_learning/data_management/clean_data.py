def clean_data(merge_individual_and_household_df):
    """Clean the merged individual and household data."""
    # Remove rows with missing values
    return merge_individual_and_household_df.dropna()
