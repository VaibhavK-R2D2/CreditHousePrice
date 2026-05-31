def load_data():
    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"

    raw_df = pd.read_csv(
        data_url,
        sep=r"\s+",
        skiprows=22,
        header=None
    )

    # Split into data and target
    data = np.hstack([
        raw_df.values[::2, :],
        raw_df.values[1::2, :2]
    ])

    target = raw_df.values[1::2, 2]

    # Feature names
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
        'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT'
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)

    # Target column
    df['MEDV'] = target

    return df