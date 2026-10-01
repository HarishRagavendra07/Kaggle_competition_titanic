from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import TitanicFeatures, deck_group, extract_title

DATA = "data/train.csv"
FIXTURE = Path(__file__).parent / "fixtures" / "passengers.csv"

# The Kaggle CSVs aren't committed (competition rules), so CI only has the synthetic fixture
needs_kaggle_data = pytest.mark.skipif(
    not (Path(DATA).exists() and Path("data/test.csv").exists()),
    reason="Kaggle data not downloaded into data/",
)


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


@needs_kaggle_data
def test_real_data_has_no_missing_values_after_transform():
    train = pd.read_csv(DATA)
    test = pd.read_csv("data/test.csv")
    features = TitanicFeatures().fit(train)
    for df in (train, test):  # test.csv has a missing Fare the training data doesn't
        out = features.transform(df)
        assert not out.isna().any().any()
        assert list(out.columns) == list(features.get_feature_names_out())


@needs_kaggle_data
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


@needs_kaggle_data
def test_imputation_uses_only_fit_data():
    train = pd.read_csv(DATA)
    features = TitanicFeatures().fit(train.iloc[:400])
    unseen = train.iloc[400:].assign(Age=np.nan)
    ages = features.transform(unseen)["Age"]
    learned = set(features.age_by_title_class_.dropna()) | set(features.age_by_title_.dropna())
    assert set(ages.unique()) <= learned


# --- Tests on the synthetic fixture: these run everywhere, including CI ---


def fixture():
    return pd.read_csv(FIXTURE)


def test_fixture_covers_every_missing_value_case():
    df = fixture()
    for col in ["Age", "Fare", "Cabin", "Embarked"]:
        assert df[col].isna().any(), f"fixture should have a missing {col}"


def test_fixture_transforms_without_missing_values():
    df = fixture()
    features = TitanicFeatures().fit(df)
    out = features.transform(df)
    assert not out.isna().any().any()
    assert list(out.columns) == list(features.get_feature_names_out())
    assert out.shape == (len(df), len(features.get_feature_names_out()))


def test_fixture_engineered_values():
    out = TitanicFeatures().fit(fixture()).transform(fixture())
    by_id = out.set_axis(fixture().PassengerId)
    assert by_id.loc[1, "FamilySize"] == 2 and by_id.loc[1, "IsAlone"] == 0
    assert by_id.loc[5, "IsAlone"] == 1
    assert by_id.loc[13, "FamilySize"] == 7
    assert by_id.loc[2, "Deck_ABC"] == 1 and by_id.loc[7, "Deck_DE"] == 1 and by_id.loc[11, "Deck_FG"] == 1
    assert by_id.loc[19, "HasCabin"] == 1 and by_id.loc[19, ["Deck_ABC", "Deck_DE", "Deck_FG"]].sum() == 0  # cabin "T"
    assert by_id.loc[14, "Title_Rare"] == 1 and by_id.loc[15, "Title_Rare"] == 1  # Rev, Countess
    assert by_id.loc[16, "Title_Miss"] == 1 and by_id.loc[20, "Title_Miss"] == 1  # Mlle, Ms
    assert by_id.loc[17, "Title_Mrs"] == 1  # Mme
    assert by_id.loc[17, "Embarked_S"] == 1  # missing port -> most common (S)
    assert by_id.loc[1, "LogFare"] == pytest.approx(np.log1p(7.25))


def test_fixture_missing_ages_use_title_and_class():
    df = fixture()
    by_id = TitanicFeatures().fit(df).transform(df).set_axis(df.PassengerId)
    assert by_id.loc[13, "Age"] == 2.0  # the only other 3rd-class Master is 2 years old
    assert by_id.loc[13, "AgeWasMissing"] == 1
    adult_3rd_class_men = df[(df.Pclass == 3) & df.Name.str.contains("Mr\\.")].Age.median()
    assert by_id.loc[6, "Age"] == adult_3rd_class_men


def test_missing_fare_uses_class_median_from_fit():
    df = fixture()
    features = TitanicFeatures().fit(df)
    out = features.transform(df).set_axis(df.PassengerId)
    assert out.loc[16, "LogFare"] == pytest.approx(np.log1p(features.fare_by_class_[2]))


def test_pipeline_trains_and_predicts_end_to_end():
    df = fixture()
    model = make_pipeline(TitanicFeatures(), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df, df.Survived)
    proba = model.predict_proba(df)[:, 1]
    assert proba.shape == (len(df),) and ((proba >= 0) & (proba <= 1)).all()
    assert (model.predict(df) == df.Survived).mean() > 0.7
