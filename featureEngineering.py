# Feature Engineering for the dataset
def featEngg(fileLoc):
    # Import required libraries for the function
    import pandas as pd
    
    # Read the data from the csv file
    df = pd.read_csv(fileLoc)

    # Drop columns that are not relevant to create the prediction model
    drop_columns = ['Unnamed: 0', 'cc_num', 'merchant', 'trans_num', 'unix_time', 'first', 'last', 'street', 'zip']
    df.drop(columns=drop_columns, inplace=True)

    # Convert Transaction
    df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])
    df['trans_date']=df['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    df['trans_date']=pd.to_datetime(df['trans_date'])
    df['dob']=pd.to_datetime(df['dob'])

    # Calculate Age
    trng_df["age"] = trng_df["trans_date"]-trng_df["dob"]
    trng_df["age"] = trng_df["age"].astype('timedelta64[Y]')