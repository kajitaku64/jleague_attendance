import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# =====================
# データ読み込み
# =====================
train = pd.read_csv("data/raw/train_add.csv")
test = pd.read_csv("data/raw/test.csv")

# =====================
# 特徴量エンジニアリング
# =====================

# 曜日を数値化（月=0）
# gameday: "03/17(土)" みたいな形式 → "03/17" を取り出す
train_md = train["gameday"].str.extract(r"(\d{2}/\d{2})")[0]
test_md  = test["gameday"].str.extract(r"(\d{2}/\d{2})")[0]

# year と結合して "YYYY/MM/DD" にする
train_date = pd.to_datetime(train["year"].astype(str) + "/" + train_md, format="%Y/%m/%d")
test_date  = pd.to_datetime(test["year"].astype(str) + "/" + test_md,  format="%Y/%m/%d")

train["gameday_num"] = train_date.dt.weekday
test["gameday_num"]  = test_date.dt.weekday


# 週末フラグ
train["is_weekend"] = train["gameday_num"].isin([5, 6]).astype(int)
test["is_weekend"] = test["gameday_num"].isin([5, 6]).astype(int)

# 時間帯（hour）
train["hour"] = train["time"].str[:2].astype(int)
test["hour"] = test["time"].str[:2].astype(int)

def time_zone(h):
    if h < 15:
        return 0  # 昼
    elif h < 18:
        return 1  # 夕方
    else:
        return 2  # 夜

train["time_zone"] = train["hour"].apply(time_zone)
test["time_zone"] = test["hour"].apply(time_zone)

# チーム人気度（平均観客数）
team_mean = train.groupby("home")["y"].mean()
train["home_pop"] = train["home"].map(team_mean)
test["home_pop"] = test["home"].map(team_mean)

train["away_pop"] = train["away"].map(team_mean)
test["away_pop"] = test["away"].map(team_mean)

# =====================
# 目的変数・説明変数
# =====================
y = train["y"]
drop_cols = ["y", "id", "gameday", "time"]
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=["id", "gameday", "time"])

# =====================
# カテゴリ変数処理（train + test まとめて）
# =====================
from sklearn.preprocessing import LabelEncoder

# train / test を縦に結合
all_data = pd.concat([X, X_test], axis=0, ignore_index=True)

for col in all_data.columns:
    if all_data[col].dtype == "object":
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

# 元に戻す
X = all_data.iloc[:len(X)].reset_index(drop=True)
X_test = all_data.iloc[len(X):].reset_index(drop=True)

# =====================
# 学習
# =====================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50)]
)

# =====================
# 評価
# =====================
pred_valid = model.predict(X_valid)
rmse = mean_squared_error(y_valid, pred_valid) ** 0.5
print("RMSE:", rmse)

# =====================
# 予測・提出
# =====================
pred_test = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "y": pred_test
})

submission.to_csv("submissions/submission_002.csv", index=False)
print("submission_002.csv 作成完了")
