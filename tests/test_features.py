import numpy as np
import pandas as pd
import pytest

from src.features import TitanicFeatures, deck_group, extract_title

DATA = "data/train.csv"


@pytest.mark.parametrize(
    "name, title",
    [
        ("Braund, Mr. Owen Harris", "Mr"),
        ("Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "Mrs"),
        ("Heikkinen, Miss. Laina", "Miss"),
        ("Palsson, Master. Gosta Leonard", "Master"),
        ("Aubart, Mme. Leontine Pauline", "Mrs"),
        ("Duff Gordon, Mlle. Laura", "Miss"),
        ("Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)", "Rare"),
        ("Byles, Rev. Thomas Roussel Davids", "Rare"),
        ("Weird name without title", "Rare"),
    ],
)
def test_extract_title(name, title):
    assert extract_title(name) == title


def test_deck_group():
    assert deck_group("C85") == "ABC"
    assert deck_group("E46") == "DE"
    assert deck_group("G6") == "FG"
    assert deck_group("T") == "Unknown"
    assert deck_group(np.nan) == "Unknown"


def test_real_data_has_no_missing_values_after_transform():
    train = pd.read_csv(DATA)
    test = pd.read_csv("data/test.csv")
    features = TitanicFeatures().fit(train)
    for df in (train, test):  # test.csv has a missing Fare the training data doesn't
        out = features.transform(df)
        assert not out.isna().any().any()
        assert list(out.columns) == list(features.get_feature_names_out())


def test_engineered_values():
    train = pd.read_csv(DATA)
    out = TitanicFeatures().fit(train).transform(train)
    row = out.loc[0]  # Braund, Mr. Owen Harris: 1 sibling/spouse, no cabin, embarked S
    assert row["FamilySize"] == 2 and row["IsAlone"] == 0
    assert row["Title_Mr"] == 1 and row["IsMale"] == 1 and row["HasCabin"] == 0
    assert row["LogFare"] == pytest.approx(np.log1p(7.25))
    assert out[[f"Title_{t}" for t in ["Mr", "Mrs", "Miss", "Master", "Rare"]]].sum(axis=1).eq(1).all()


def test_age_is_imputed_from_title_and_class_medians_learned_in_fit():
    train = pd.DataFrame(
        {
            "Name": ["A, Master. X", "B, Master. Y", "C, Master. Z", "D, Mr. W"],
            "Pclass": [3, 3, 3, 3],
            "Sex": ["male"] * 4,
            "Age": [2.0, 6.0, np.nan, 40.0],
            "SibSp": [0] * 4,
            "Parch": [0] * 4,
            "Fare": [10.0] * 4,
            "Cabin": [np.nan] * 4,
            "Embarked": ["S"] * 4,
        }
    )
    out = TitanicFeatures().fit(train).transform(train)
    assert out.loc[2, "Age"] == 4.0  # median of the boys in 3rd class, not the adult's 40
    assert out.loc[2, "AgeWasMissing"] == 1


def test_imputation_uses_only_fit_data():
    train = pd.read_csv(DATA)
    features = TitanicFeatures().fit(train.iloc[:400])
    unseen = train.iloc[400:].assign(Age=np.nan)
    ages = features.transform(unseen)["Age"]
    learned = set(features.age_by_title_class_.dropna()) | set(features.age_by_title_.dropna())
    assert set(ages.unique()) <= learned
