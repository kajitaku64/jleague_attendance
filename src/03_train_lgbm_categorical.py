import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv("data/raw/train_add.csv")
test = pd.read_csv("data/raw/test.csv")

# =====================
# stadium.csv を結合
# =====================
stadium = pd.read_csv("data/raw/stadium.csv")

train = train.merge(
    stadium,
    left_on="stadium",
    right_on="name",
    how="left"
)

test = test.merge(
    stadium,
    left_on="stadium",
    right_on="name",
    how="left"
)

condition = pd.read_csv("data/raw/condition.csv")

train = train.merge(condition, on="id", how="left")
test  = test.merge(condition,  on="id", how="left")

# weather: "雨", "晴", "曇" などを想定
train["is_rain"] = train["weather"].astype(str).str.contains("雨").astype(int)
test["is_rain"]  = test["weather"].astype(str).str.contains("雨").astype(int)

train["temperature"] = train["temperature"]
test["temperature"]  = test["temperature"]

train["humidity"] = train["humidity"]
test["humidity"]  = test["humidity"]

# 予測時に使えない情報（リーク防止）
drop_leak_cols = [
    "home_score", "away_score",
    "home_team", "away_team",
    "home_01", "home_02", "home_03", "home_04", "home_05",
    "home_06", "home_07", "home_08", "home_09", "home_10", "home_11",
    "away_01", "away_02", "away_03", "away_04", "away_05",
    "away_06", "away_07", "away_08", "away_09", "away_10", "away_11",
    "referee"
]

train = train.drop(columns=[c for c in drop_leak_cols if c in train.columns])
test  = test.drop(columns=[c for c in drop_leak_cols if c in test.columns])


# ---------- feature engineering ----------
# gameday: "03/17(土)" -> "03/17" 抽出して year と結合
train_md = train["gameday"].str.extract(r"(\d{2}/\d{2})")[0]
test_md  = test["gameday"].str.extract(r"(\d{2}/\d{2})")[0]

train_date = pd.to_datetime(train["year"].astype(str) + "/" + train_md, format="%Y/%m/%d")
test_date  = pd.to_datetime(test["year"].astype(str) + "/" + test_md,  format="%Y/%m/%d")

train["weekday"] = train_date.dt.weekday
test["weekday"]  = test_date.dt.weekday

train["is_weekend"] = train["weekday"].isin([5, 6]).astype(int)
test["is_weekend"]  = test["weekday"].isin([5, 6]).astype(int)

train["hour"] = train["time"].str[:2].astype(int)
test["hour"]  = test["time"].str[:2].astype(int)

def time_zone(h):
    if h < 15: return "day"
    if h < 18: return "evening"
    return "night"

train["time_zone"] = train["hour"].apply(time_zone)
test["time_zone"]  = test["hour"].apply(time_zone)

# team popularity
team_mean = train.groupby("home")["y"].mean()
train["home_pop"] = train["home"].map(team_mean)
test["home_pop"]  = test["home"].map(team_mean)
train["away_pop"] = train["away"].map(team_mean)
test["away_pop"]  = test["away"].map(team_mean)

# ---------- X, y ----------
y = train["y"]
drop_cols = ["y", "id", "gameday", "time"]
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=["id", "gameday", "time"])

# =====================
# 欠損値処理
# =====================
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].fillna("Unknown")
        X_test[c] = X_test[c].fillna("Unknown")
    else:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

# カテゴリ列を category 型にする（LabelEncoder不要）
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
for c in cat_cols:
    all_cats = pd.concat([X[c], X_test[c]], axis=0).astype("category")
    X[c] = all_cats.iloc[:len(X)].reset_index(drop=True)
    X_test[c] = all_cats.iloc[len(X):].reset_index(drop=True)

# split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(100)]
)

pred_valid = model.predict(X_valid)
rmse = mean_squared_error(y_valid, pred_valid) ** 0.5
print("RMSE:", rmse)

pred_test = model.predict(X_test)

submission = pd.DataFrame({"id": test["id"], "y": pred_test})
submission.to_csv("submissions/submission_003.csv", index=False)
print("submission_003.csv 作成完了")
