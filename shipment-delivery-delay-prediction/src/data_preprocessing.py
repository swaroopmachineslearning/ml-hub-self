import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    df['ShipmentStartTime'] = pd.to_datetime(df['ShipmentStartTime'])
    df['ShipmentWeekday'] = df['ShipmentStartTime'].dt.weekday
    df['ShipmentHour'] = df['ShipmentStartTime'].dt.hour

    df.drop(['ShipmentID','ShipmentStartTime'],axis=1,inplace=True)

    #Encode Categoricals (string object to ML formed data)
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])

    X = df.drop('IsDelayed', axis=1)
    y = df['IsDelayed']
    return train_test_split(X, y, test_size=0.2, random_state=42)
