import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv', sep='\t')

sensorid = pd.DataFrame(data["SensorID"])

for i in range(len(sensorid)):
    a = sensorid['SensorID'][i]
    if a[0] == 'M':
        a = '1' + a[1:]
    elif a[0] == 'I':
        a = '2' + a[1:]
    elif a[0] == 'D':
        a = '3' + a[1:]
    data.loc[i, 'SensorID'] = a


sensorvalue = pd.DataFrame(data["Value"])

for i in range(len(sensorvalue)):
    a = sensorvalue['Value'][i]
    if a == 'ON':
        a = '1'
    elif a == 'OFF':
        a = '0'
    elif a == 'OPEN':
        a = '3'
    elif a == 'CLOSE':
        a = '4'
    elif a == 'PRESENT':
        a = '5'
    elif a == 'ABSENT':
        a = '6'
    data.loc[i, 'Value'] = a

df = pd.DataFrame(columns=["Time", "SensorID", "Value", "TaskID","ResidentID"])
df["Time"] =data["Time"]
df["SensorID"] = data["SensorID"]
df["Value"] = data["Value"]
df["ResidentID"] = data["ResidentID"]
df["TaskID"] = data["TaskID"]

print(df)


def encode_time(time_str):
    # Split the time string into hours, minutes, seconds, and microseconds
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds, microseconds = map(int, parts[2].split('.'))  # Split seconds and microseconds

    # Scale each feature
    scaled_hours = hours / 24  # Scale hours to [0, 1]
    scaled_minutes = minutes / 60  # Scale minutes to [0, 1]
    scaled_seconds = seconds / 60  # Scale seconds to [0, 1]
    scaled_microseconds = microseconds / 1e6  # Scale microseconds to [0, 1]

    # Calculate the fraction of the day
    fraction_of_day = (scaled_hours + scaled_minutes / 60 + scaled_seconds / 3600 +
                       scaled_microseconds / 3600 )

    # Convert to string
    fraction_str = str(fraction_of_day)

    return fraction_str


for i in range(len(df["Time"])):
    df.loc[i, 'Time'] = encode_time(df["Time"][i])

df.to_csv('df.csv', sep='\t', index=False)