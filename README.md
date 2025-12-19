# Jリーグ観客動員数予測（SIGNATE 練習問題）

## 概要
本プロジェクトは、SIGNATE の練習問題「Jリーグの観客動員数予測」に取り組んだ記録である。  
過去の試合データ（開催条件・クラブ・スタジアム・中継情報など）を用いて、各試合の観客動員数を予測する回帰モデルを構築した。

## データ
- train_add.csv：学習用データ（観客動員数 y を含む）
- test.csv：予測対象データ
- stadium.csv：スタジアム情報（収容人数など）

## 特徴量エンジニアリング
- クラブ人気：home_pop / away_pop
- スタジアム要因：capa（収容人数）
- 開催条件：曜日・時間帯・ナイトゲーム判定
- 注目度：tv_count、NHK中継フラグ

## モデル・評価
- モデル：LightGBM Regressor
- 評価指標：RMSE
- 学習方法：5-Fold Cross Validation

主なハイパーパラメータ：
- n_estimators: 8000
- learning_rate: 0.03
- num_leaves: 31
- subsample / colsample_bytree: 0.9
- early stopping: 200 rounds

カテゴリ変数は LightGBM の categorical feature 機能を利用した。

## 学んだこと
- 回帰タスクにおける 特徴量設計の重要性
- KFold による汎化性能評価の有効性
- 「重要度が0の特徴量でも、CVが改善する場 合がある」こと
- モデル精度だけでなく、解釈可能性を意識することの大切さ
