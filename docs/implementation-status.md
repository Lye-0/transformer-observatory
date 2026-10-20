# 初期リリースの実装状況

仕様書v1.0の初期リリース優先範囲を実装し、小型モデルで検証した。対応範囲と拡張機能を混同しない。

| 領域 | 実装と確認 |
|---|---|
| 基盤 | Python/Node/pnpm固定、Windows/WSL別環境、セットアップ・起動、CI |
| 取得 | 全Residual、選択Layer/Head/テンソル、Qwen3/GPT-2構造、独立参照・実行時再構成チェック |
| 射影・指標 | 固定共通PCA、Scientific/Displacement、元高次元の距離・角度、未定義値の区別 |
| 3D | 全軌跡のバッチ描画、検索・複数選択、カメラ、Layer/Decodeタイムライン、成長表示・明るさ・更新矢印・軽量表示 |
| Inspector | Attention/QKV/GQA/Mask/Head接続、MLP、Tensor値・ヒートマップ・統計、Pin、未取得の明示と追加解析の入力導線 |
| ジョブ | 6工程、概算進捗、中止・失敗・部分データ、再接続、バックエンドによる継続 |
| 保存 | Level 1〜4、NPY逐次保存、所有・ハッシュ・位置/形状検証、保持・削除・一括整理、ZIP入出力、実験メモとタグ |
| 比較・生成 | 両実験からの共通PCA、Overlay/Side by Side/Difference、全Token対応表、共通先頭検証、KV Cacheの追加位置記録 |
| 実モデル | Qwen3-0.6BのWindows/WSL CUDA実測。GPT-2は小型ランダム構成のアダプター検証まで |
| 予測読出し | Logit Lens。最終Norm/LM Headの読み出しと明記 |

仕様のPhase番号はこの検証を順次進めるために使い、途中でユーザー操作待ちとして作業を止める基準にはしていない。
Phase 6の任意サイズ・量子化・長文最適化と、Phase 7のTuned Lens・介入実験・追加射影は拡張項目。初期リリースでの対応を保証しない。
観察値と因果性を区別し、取得できない値や未学習のLensを生成して対応済みと表示しない。

## 検証資料

- [数値・両OS GPU・ブラウザ検証](numerical-validation.md)
- [モデル別の取得範囲](model-support.md)
- [永続形式と削除の契約](trace-format.md)
- [環境とGitHub経由のWSL検証手順](development-environment.md)

既知の範囲: 内部取得はEager PyTorchのみ、ジョブは直列。テンソルのブラウザ表示は先頭16,384成分だが、保存値と主軌跡は削減しない。
インポート・比較・Logit Lensの読み出しは大容量データでは時間がかかり得る。PreflightのRAM/VRAM値は概算で、任意サイズでの実行可能性を保証しない。
