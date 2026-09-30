"""Feature engineering for the Titanic dataset.

`TitanicFeatures` is a scikit-learn transformer, so every value it fills in (median ages by
Title and class, the fare median, the Embarked mode) is learned in `fit()` from the training
data only. Inside a Pipeline, cross-validation re-learns them on each training fold, which
keeps the test folds from leaking into the model.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLES = ["Mr", "Mrs", "Miss", "Master", "Rare"]
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
DECK_GROUPS = {"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE", "F": "FG", "G": "FG"}
EMBARKED = ["C", "Q", "S"]


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Uncommon titles (Dr, Rev, Countess...) become 'Rare'."""
    title = name.split(",", 1)[1].split(".", 1)[0].strip() if "," in name and "." in name else ""
    title = TITLE_ALIASES.get(title, title)
    return title if title in TITLES else "Rare"


def deck_group(cabin) -> str:
    """First letter of the cabin, grouped by position on the ship; 'T' and missing -> 'Unknown'."""
    if not isinstance(cabin, str) or not cabin:
        return "Unknown"
    return DECK_GROUPS.get(cabin[0], "Unknown")


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Raw Kaggle columns in, numeric model-ready features out."""

    def fit(self, X: pd.DataFrame, y=None):
        df = X.assign(Title=X["Name"].map(extract_title))
        self.age_by_title_class_ = df.groupby(["Title", "Pclass"])["Age"].median()
        self.age_by_title_ = df.groupby("Title")["Age"].median()
        self.age_median_ = df["Age"].median()
        self.fare_by_class_ = df.groupby("Pclass")["Fare"].median()
        self.embarked_mode_ = df["Embarked"].mode()[0]
        return self

    def _impute_age(self, df: pd.DataFrame) -> pd.Series:
        by_group = pd.Series(
            [self.age_by_title_class_.get((t, c), np.nan) for t, c in zip(df["Title"], df["Pclass"])],
            index=df.index,
        )
        fallback = df["Title"].map(self.age_by_title_).fillna(self.age_median_)
        return df["Age"].fillna(by_group).fillna(fallback)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.assign(Title=X["Name"].map(extract_title))
        fare = df["Fare"].fillna(df["Pclass"].map(self.fare_by_class_))
        embarked = df["Embarked"].fillna(self.embarked_mode_)
        family = df["SibSp"] + df["Parch"] + 1
        deck = df["Cabin"].map(deck_group)

        out = pd.DataFrame(index=X.index)
        out["Pclass"] = df["Pclass"]
        out["IsMale"] = (df["Sex"] == "male").astype(int)
        out["Age"] = self._impute_age(df)
        out["AgeWasMissing"] = df["Age"].isna().astype(int)
        out["FamilySize"] = family
        out["IsAlone"] = (family == 1).astype(int)
        out["LogFare"] = np.log1p(fare)
        out["HasCabin"] = df["Cabin"].notna().astype(int)
        for group in ["ABC", "DE", "FG"]:
            out[f"Deck_{group}"] = (deck == group).astype(int)
        for title in TITLES:
            out[f"Title_{title}"] = (df["Title"] == title).astype(int)
        for port in EMBARKED:
            out[f"Embarked_{port}"] = (embarked == port).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["Pclass", "IsMale", "Age", "AgeWasMissing", "FamilySize", "IsAlone", "LogFare", "HasCabin"]
            + [f"Deck_{g}" for g in ["ABC", "DE", "FG"]]
            + [f"Title_{t}" for t in TITLES]
            + [f"Embarked_{p}" for p in EMBARKED]
        )
