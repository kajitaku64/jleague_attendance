import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# =====================
# データ読み込み
# =====================
train = pd.read_csv("data/train_add.csv")
test = pd.read_csv("data/test.csv")

# =====================
# 目的変数と特徴量
# =====================
y = train["y"]
X = train.drop(columns=["y", "id"])
test_id = test["id"]
X_test = test.drop(columns=["id"])

# =====================
# カテゴリ変数を数値化
# =====================
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_test[col] = le.transform(X_test[col])

# =====================
# 学習・検証データ分割
# =====================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# LightGBM 学習
# =====================
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
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
# 検証RMSE
# =====================
pred_valid = model.predict(X_valid)
rmse = mean_squared_error(y_valid, pred_valid, squared=False)
print("RMSE:", rmse)

# =====================
# テストデータ予測
# =====================
pred_test = model.predict(X_test)

# =====================
# 提出ファイル作成
# =====================
submission = pd.DataFrame({
    "id": test_id,
    "y": pred_test
})

submission.to_csv("submissions/submission_001.csv", index=False)
print("submission_001.csv を作成しました")

