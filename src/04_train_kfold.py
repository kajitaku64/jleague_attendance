import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# =====================
# load
# =====================
train = pd.read_csv("data/raw/train_add.csv")
test  = pd.read_csv("data/raw/test.csv")
test_ids = test["id"].copy()   # ★追加：元のtestのidを保存

# =====================
# stadium merge
# =====================
stadium = pd.read_csv("data/raw/stadium.csv")  # columns: name, address, capa
train = train.merge(stadium, left_on="stadium", right_on="name", how="left")
test  = test.merge(stadium,  left_on="stadium", right_on="name", how="left")

# nameは重複なので削除（addressは残してもOK）
train = train.drop(columns=["name"])
test  = test.drop(columns=["name"])

# =====================
# feature engineering
# =====================

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
train["is_night"] = (train["hour"] >= 18).astype(int)
test["is_night"]  = (test["hour"] >= 18).astype(int)

# team popularity
team_mean_home = train.groupby("home")["y"].mean()
train["home_pop"] = train["home"].map(team_mean_home)
test["home_pop"]  = test["home"].map(team_mean_home)
train["away_pop"] = train["away"].map(team_mean_home)
test["away_pop"]  = test["away"].map(team_mean_home)

# ---- 追加①：stage（節）を数値化（例: "第３４節第２日" -> 34） ----
train["stage_round"] = train["stage"].astype(str).str.extract(r"第(\d+)節")[0].astype(float)
test["stage_round"]  = test["stage"].astype(str).str.extract(r"第(\d+)節")[0].astype(float)

# ---- 追加②：tv を分解（中継数・NHKフラグ） ----
train_tv = train["tv"].astype(str)
test_tv  = test["tv"].astype(str)

train["tv_count"] = train_tv.str.count("／") + 1
test["tv_count"]  = test_tv.str.count("／") + 1

train["is_nhk"] = train_tv.str.contains("ＮＨＫ").astype(int)
test["is_nhk"]  = test_tv.str.contains("ＮＨＫ").astype(int)

# =====================
# X, y
# =====================
y = train["y"]
drop_cols = ["y", "id", "gameday", "time"]
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=["id", "gameday", "time"])


# =====================
# missing values
# =====================
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].fillna("Unknown")
        X_test[c] = X_test[c].fillna("Unknown")
    else:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

# =====================
# categorical handling
# =====================
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
for c in cat_cols:
    all_c = pd.concat([X[c], X_test[c]], axis=0).astype("category")
    X[c] = all_c.iloc[:len(X)].reset_index(drop=True)
    X_test[c] = all_c.iloc[len(X):].reset_index(drop=True)

# =====================
# KFold training
# =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

feature_names = X.columns.tolist()
imp_sum = pd.Series(0.0, index=feature_names)

oof_rmse = []
pred_test = 0.0

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(200)]
    )

    imp_sum += pd.Series(model.feature_importances_, index=feature_names)

    pred_va = model.predict(X_va)
    rmse = mean_squared_error(y_va, pred_va) ** 0.5
    oof_rmse.append(rmse)
    print(f"fold {fold} rmse: {rmse:.4f}")

    pred_test += model.predict(X_test) / kf.n_splits

print("CV RMSE:", sum(oof_rmse) / len(oof_rmse))

imp_mean = (imp_sum / kf.n_splits).sort_values(ascending=False)
print("\n=== Feature Importance (mean over folds) ===")
print(imp_mean)
import os
os.makedirs("outputs", exist_ok=True)

imp_mean.to_csv("outputs/feature_importance_mean.csv")
print("\nfeature_importance_mean.csv を outputs/ に保存しました")

# =====================
# submission
# =====================
# =====================
# submission
# =====================
print("len(test_ids) =", len(test_ids))
print("len(X_test)   =", len(X_test))
print("len(pred_test)=", len(pred_test))
assert len(pred_test) == len(test_ids), "pred_testの長さがtest_idsと一致しません（行が落ちてる可能性）"

sub = pd.DataFrame({"id": test_ids.values, "y": pred_test})
sub.to_csv("submissions/submission_005.csv", index=False)
print("submission_005.csv 作成完了")
