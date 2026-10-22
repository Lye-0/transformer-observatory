# 開発環境と合意事項

更新: 2026-10-20。提供仕様書の原文は `Transformer_Observatory_Development_Specification.md` に保存し、後続のユーザー指示をこの文書へ記録する。

2026-10-22追記: WindowsのPythonを同じ3.13系列の3.13.17へ更新。セットアップスクリプトとCIの採用版も揃えた。WSL上の既存環境は今回更新・再検証していない。以下に記載した初期採用の経緯と過去の検証成果物の版は履歴として維持する。

## 配置

- Windows作業clone: `C:\Users\kawau\dev\LLM\transformer-observatory`
- WSL作業clone（ユーザー承認済み）: `/home/lye/dev/llm/transformer-observatory`
- WSL既存参考環境: `/home/lye/dev/llm/llm-lab`。本プロジェクトの依存環境として再利用せず、変更しない。
- リモート: `https://github.com/Lye-0/transformer-observatory.git`

Windowsで変更・テスト・コミットしてGitHubへpushし、WSLの独立cloneで `git pull --ff-only` して検証する。
仮想環境・node_modulesはOSごとに作る。Windowsの `.venv` をWSLから実行しない。共有するのはソース・ロックファイル・ポータブルなトレース。

## バージョン

| 対象 | 版 | 管理 |
|---|---|---|
| Python | 3.13.17 | uv、`.python-version`、`backend/.venv` |
| Node.js | 24.21.0 | ルートのmise.toml |
| pnpm | 11.24.0 | mise.tomlとpackageManager |
| PyTorch | 2.14.1+cu130 | backend/uv.lock、公式cu130 index |
| Transformers | 5.18.0 | backend/uv.lock |

Pythonは仕様書の3.12提案に代わり、ユーザーの希望でWSL既存環境と同じ3.13.15を採用した。グローバルPython 3.11を置き換えない。
初期構築時はプロジェクト用Python本体・uvキャッシュを `.tools/` に配置した。2026-10-22の整理でPython本体とキャッシュはuvのユーザーごとの既定保存先へ統一。Windows仮想環境は `backend/.venv` を維持する。
WSLのmise本体はユーザーが導入済み。Node・pnpm・Python・依存関係を専用cloneへ導入し、両OSのビルド・CPU/CUDA検証を実施した。

## モデル

数値検証用の極小Transformerに加え、Qwen3-0.6BとGPT-2 Smallの実重みを両OSで検証した記録がある。大規模モデル実験は現在の対象外。
モデル重みはGitへ追加せず、OSごとに保存場所を登録する。2026-10-20の両OS検証ではfingerprintの一致を確認した。
`.tools/models/` はモデルの配置候補であり、現在もファイルが存在することを保証しない。2026-10-21の確認ではWindowsの従来のQwen3-0.6B／GPT-2配置先は存在しなかった。
現在の配置・対応範囲はModel Managerで確認する。標準キャッシュ・任意の探索先の検出、パス入力、フォルダー選択と再指定は[ローカルモデルの登録](model-registration.md)を参照。

## このPCで確認した点

Windows 11 Home / i7-13700K / RAM約96GB / RTX 4080 SUPER 16GB、NVIDIAドライバー616.56。WSL Ubuntu 26.04 LTS。
Windows CPUのスレッド数とCUDA積和精度の調整理由は [数値検証](numerical-validation.md) を参照。
初期構築時はuvの既定保存先で失敗したため、プロジェクト内のPython/キャッシュ先を指定していた。現在はこの上書きを廃止し、通常ユーザーのuv既定保存先を使用する。Windowsで確認したPython本体の保存先は `C:\Users\kawau\AppData\Roaming\uv\python`。Codex固有LocalCacheやCodex付属Pythonを外部プロジェクトから参照しない。WSLもセットアップスクリプト上はuv既定保存先を使用するが、今回WSLの既存環境は変更・再検証していない。
PowerShellのmise関数による引数解釈の影響を避け、Windowsスクリプトでは `mise.exe` を呼ぶ。
Windows/WSLの仮想ディスクは同じ物理SSDを使うため、空き容量はそれぞれのバックエンドで確認する。

## セットアップと更新

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
```

```bash
cd /home/lye/dev/llm/transformer-observatory
git pull --ff-only
bash scripts/setup.sh
bash scripts/start.sh
```

ロックファイルの内容からセットアップする。依存変更がなければ、ソース更新後にフロントエンドをビルドしバックエンドを再起動すればよい。
フェーズ完了は作業の停止条件にしない。追加ユーザー操作が必要な場合のみ相談する。
