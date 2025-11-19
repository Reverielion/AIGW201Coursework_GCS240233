import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PredictForest:
    def __init__(self):
        self.df = pd.read_csv(r"C:\Users\Asus\PycharmProjects\CourseworkAI\Week 4 - dataset_2_advertising.csv", index_col=0)
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.fillna(self.df.select_dtypes(include='number').mean(), inplace=True)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        self.ct = ColumnTransformer([('num', StandardScaler(), self.cols)])
        self.X = self.ct.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        self.model = RandomForestRegressor(n_estimators=200, random_state=1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def retrain(self, path):
        self.df = pd.read_csv(path, index_col=0)
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.fillna(self.df.select_dtypes(include='number').mean(), inplace=True)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        self.ct = ColumnTransformer([('num', StandardScaler(), self.cols)])
        self.X = self.ct.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        self.model = RandomForestRegressor(n_estimators=200, random_state=1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def manual_predict_input(self, inputstring):
        try:
            features = [float(x) for x in inputstring.strip().split()]
            if len(features) != self.X.shape[1]:
                return f"Invalid input. Make sure input suits dataset and separated by space."
            data = pd.DataFrame([features], columns=self.df.columns[:-1])
            data_scaled = self.ct.transform(data)
            y_pred_new = self.model.predict(data_scaled)
            return y_pred_new[0]
        except Exception as e:
            return f"Invalid input. Make sure input suits dataset and separated by space."

    def imageplot(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.y_test, self.y_pred)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)],linestyle='--', linewidth=2)
        plt.title("Test vs Prediction (Random Forest)")
        plt.xlabel("Actual")
        plt.ylabel("Prediction")
        plt.tight_layout()
        plt.savefig("predictiontest.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    model = PredictForest()
    # model.retrain(r"C:\Users\Asus\PycharmProjects\CourseworkAI\Week 5 - housing_dataset.csv")
    inputtext = input()
    print(model.manual_predict_input(inputtext))