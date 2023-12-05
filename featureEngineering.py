# Feature Engineering for the dataset
def featEngg(fileLoc, training_df):
    # Import required libraries for the function
    import pandas as pd
    
    # Read the data from the csv file
    df = pd.read_csv(fileLoc)

    # Drop columns that are not relevant to create the prediction model
    drop_columns = ['Unnamed: 0', 'cc_num', 'merchant', 'trans_num', 'unix_time', 'first', 'last', 'street', 'zip']
    df.drop(columns=drop_columns, inplace=True)

    # Handle Transaction date and time column from the dataset
    df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])
    df['trans_date']=df['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    df['trans_date']=pd.to_datetime(df['trans_date'])
    df['trans_month'] = pd.DatetimeIndex(df['trans_date']).month
    df['trans_year'] = pd.DatetimeIndex(df['trans_date']).year

    # Calculate Age
    df['dob']=pd.to_datetime(df['dob'])
    df["age"] = df["trans_date"]-df["dob"]
    df["age"] = df["age"].astype('timedelta64[Y]')

    # Calculate distance between the merchant and the transaction location
    df['latitudinal_distance'] = abs(round(df['merch_lat']-df['lat'],3))
    df['longitudinal_distance'] = abs(round(df['merch_long']-df['long'],3))

    # Drop the rest of the columsn that are irrelevant after computational changes to the dataset
    drop_add_columns = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','trans_date','state']
    df.drop(columns=drop_add_columns,inplace=True)

    # Convert categorical column 'gender' to numerical column
    df.gender = df.gender.apply(lambda x: 1 if x=="M" else 0)

    # Convert categorical column category to numerical column
    df = pd.get_dummies(df, columns=['category'], prefix='category')

    df = df.reindex(columns=training_df.columns, fill_value=0)
    return df