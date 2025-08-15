import pandas as pd
df = pd.read_csv('../Dataset/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

def X_y():
        return X, y


if __name__ == '__main__':
        X_y()
