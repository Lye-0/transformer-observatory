# Transformer Observatory

LLMのトークン位置に対応する内部ベクトルを、実際のモデル演算から取得して観測するローカル研究ツール。
推論・数値計算・保存はPython、3D操作はlocalhostのブラウザで行います。

## 起動

Windows PowerShell:

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
```

WSL / Linux（Linux側に独立clone）:

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

Python環境は `backend/.venv`、uv管理Pythonとキャッシュは `.tools/` に配置します。
uvとmiseが必要です。Python 3.13.15、Node.js 24.21.0、pnpm 11.24.0を使用します。
セットアップには依存関係のダウンロードが必要です。モデルはユーザー指定のローカルsafetensorsを読み込みます。
起動時の既定ポートは8765。使用中なら空きポートを使い、そのURLを表示します。
Ctrl+Cでバックエンドを終了します。ブラウザを閉じてもバックエンド内のジョブは継続します。

## 最初の観測

1. 「新しい解析」で数値検証用の極小Transformerを選び、入力して開始します。
2. 実モデルはModel Managerでconfig.json・Tokenizer・safetensorsのあるフォルダーを登録します。
3. 3D ExplorerでScientific / Displacement、トークン位置、Layer・Attention後／MLP後を操作します。
4. Level 3/4で取得したAttention・MLPの値をInspectorから調べます。
5. Trace Libraryから保持・削除・エクスポート・インポートできます。
6. 同一モデルのLevel 2以上の実験は、両者から計算した共通PCAで比較できます。
7. ResearchでResidual Patch・Head/MLP Ablation、複数実験の記述統計、全点t-SNEを実行し、条件付きJSON・CSVを出力できます。

実モデルは「モデルを追加」から検出・絶対パス入力・フォルダー選択で登録します。
標準のモデル置き場とHugging Faceキャッシュを探索し、追加の探索先も保存できます。登録後は入力途中のプロンプトを保って解析画面に戻ります。
モデル重みの自動ダウンロードは行いません。モデルファイルの配置状況はModel Managerで確認してください。
操作・検出範囲・移動後の再指定は[ローカルモデルの登録](docs/model-registration.md)を参照してください。
新しい実験はLevel 2が既定。特定Blockを調べるときはLevel 3/4と対象Layer・Head・テンソル名を指定します。

極小Transformerは固定ランダム重みの実計算モデルです。事前学習済みLLMではありません。
内部値が未取得の場合は未取得と表示します。投影された距離やAttention係数を因果的重要性として扱いません。

## 保存

Windowsの既定一時トレースは `%TEMP%/TransformerObservatory/traces/`。
永続設定・実験一覧は `%LOCALAPPDATA%/TransformerObservatory/`。
LinuxはXDGのdata/cacheディレクトリを使用します。
`OBSERVATORY_DATA_DIR` と `OBSERVATORY_TRACE_DIR` で起動時に変更できます。
保存先はバックエンドOS上の絶対パスです。
保持した実験はカタログ配下の `retained/traces` へコピー・照合後に移します。
設定画面では新しいトレースの保存先を変更できます。実験名・説明・タグはLibraryから編集できます。
同じデータフォルダーを使うバックエンドの二重起動は拒否します。

トレースはversion付きmanifest・所有マーカー・NPYチャンク・SHA-256で保存します。
NumPyのpickle読込は使いません。不要な一時データはLibraryのプレビューを確認して削除します。
Level 1でも射影確定までは生Residualの一時容量が必要です。

## 開発・検証

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests -q
mise.exe exec -- pnpm --dir frontend run build
```

```bash
backend/.venv/bin/python -m pytest backend/tests -q
mise exec -- pnpm --dir frontend run build
```

- [環境方針](docs/development-environment.md)
- [実装契約](docs/architecture.md)
- [総合仕様書](docs/Transformer_Observatory_Development_Specification.md)
- [数値検証](docs/numerical-validation.md)
- [モデル対応](docs/model-support.md)
- [トレース形式](docs/trace-format.md)
- [実装範囲と拡張機能](docs/implementation-status.md)
- [介入・統計・追加射影・Tuned Lens](docs/research.md)

Qwen3-0.6Bは両OSのCUDAで、全28層・2トークン生成・Deep Traceを検証済みです。
各入力756項目の数値チェックを通過しています。実測時間・メモリ・再現手順は数値検証文書に記載しています。
GPT-2 Smallも両OSのCUDAで実重みを検証し、公式公開Tuned Lensの読み出しを確認済みです。量子化は非対応です。
Tuned Lensは対象モデルに合う学習済みローカル重みを登録すると利用できます。GPT-2用の公開学習済み重みで実装を検証済みです。Qwen3-0.6B用は検証時にダウンロードできる学習済み重みを確認できておらず、同モデルでのLens予測品質は未検証です。

localhost APIはHost・Originを検証し、セッションと操作トークンを要求します。
インターネット向けサーバーとして公開しないでください。
