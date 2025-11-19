import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

class NaiveBayesClass:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = pd.DataFrame(self.df)
        self.target_column = self.df.columns[-1]
        self.encoders = {}
        for col in self.df.columns:
            if self.df[col].dtype == "object" or self.df[col].dtype.name == "category":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        self.model = CategoricalNB()
        self.model.fit(self.X, self.y)

    def encode_sample(self, sample_df):
        df = sample_df.copy()
        for col in df.columns:
            if col in self.encoders:
                df[col] = self.encoders[col].transform(df[col])
        return df

    def predict(self, path):
        samples = pd.read_csv(path)
        if isinstance(samples, dict):
            samples = [samples]

        sample_df = pd.DataFrame(samples)
        sample_encoded = self.encode_sample(sample_df)

        preds = self.model.predict(sample_encoded)
        probs = self.model.predict_proba(sample_encoded)

        if self.target_column in self.encoders:
            preds = self.encoders[self.target_column].inverse_transform(preds)
            prob_columns = self.encoders[self.target_column].classes_
        else:
            prob_columns = self.model.classes_
        probs_df = pd.DataFrame(probs, columns=prob_columns)
        return preds, probs_df



# ------------------------
# Usage
# ------------------------
if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\Asus\Documents\Coursework_IntroAI\class-train.csv")
    model = NaiveBayesClass(data)
    samples = pd.read_csv(r"C:\Users\Asus\Documents\Coursework_IntroAI\class-test.csv")
    #preds, probs = model.predict(samples)
    preds = model.predict(samples)
    print("Predictions:\n", preds)
    #print("\nProbabilities:\n", probs)

