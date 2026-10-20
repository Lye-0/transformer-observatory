# 実装契約と技術計画

仕様書と2026-10-20の追加合意に従い、Windowsを優先しWSLを独立した実行環境として検証する。

## 責務

- Python 3.13.15、PyTorch、Transformers: 実モデル推論、取得、数値検証。
- FastAPI: loopback API、セッション認証、ジョブ、ストレージ制御。
- React/TypeScript/Three.js: 全トークンの3D軌跡、タイムライン、Inspector。
- SQLite: 設定、モデル登録、ジョブとトレースの検索。
- トレース: version付きmanifest、所有マーカー、SHA-256、観測点単位のNPYチャンク。

NPYを初期保存形式とする。各観測点を逐次同期書き込みし、GPU履歴と無制限RAMキューを持たない。
読取時はmmapで部分参照可能。Zarrへの依存を必須にせず、フォーマット移行はschema_versionで扱う。
Level 1はPCA確定までResidualを一時スプールし、正常完了時に生テンソルを破棄する。

## 検証順序

1. 決定論的な極小SwiGLU Transformerと独立NumPy参照計算。
2. Residual/Attention/MLP再構成、固定PCA、変位と指標、保存と削除。
3. API・失敗パス・ジョブ再接続、3D表示と実際のUI操作。
4. Qwen3-0.6BのWindows/WSL GPU検証、比較と生成。

段階完了は停止条件ではない。実測結果と未対応項目をmodel-support.mdおよび検証文書へ記録する。

## モデルと空間

極小Transformerはランダム固定重みの実計算を行うテストモデルであり、事前学習済みLLMとして表示しない。
Qwen3とGPT-2は構造別アダプターで扱う。未対応アーキテクチャや量子化モデルの登録を拒否する。
Residual、Q/K/V、MLP中間空間を識別し、Residual主軌跡にはResidualに戻した更新のみ表示する。
Qwen3のQ/KはQK NormとRoPEを区別し、GQAのQuery→KV Head対応を保存する。

## 制約

対応モデル、最大長、PCA性能、同時ジョブ性能は実測で確定する。初期の推論ジョブは直列実行する。
共有PCAは実験内で固定し、比較時は両トレースから共通基底を再計算する。
未知のバックエンドで内部値を取得可能と表示しない。介入実験は観察と明確に分ける。
