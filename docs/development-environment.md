# 開発環境と合意事項

更新: 2026-10-20。提供仕様書の原文は `Transformer_Observatory_Development_Specification.md` に保存し、後続のユーザー指示をこの文書へ記録する。

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
| Python | 3.13.15 | uv、`.python-version`、`backend/.venv` |
| Node.js | 24.21.0 | ルートのmise.toml |
| pnpm | 11.24.0 | mise.tomlとpackageManager |
| PyTorch | 2.14.1+cu130 | backend/uv.lock、公式cu130 index |
| Transformers | 5.18.0 | backend/uv.lock |

Pythonは仕様書の3.12提案に代わり、ユーザーの希望でWSL既存環境と同じ3.13.15を採用した。グローバルPython 3.11を置き換えない。
プロジェクト用Python本体・uvキャッシュは `.tools/`、Windows仮想環境は合意済みの `backend/.venv` に作成済み。
WSLのmise本体はユーザーが導入済み。Node・pnpm・Python・依存関係を専用cloneへ導入し、両OSのビルド・CPU/CUDA検証を実施した。

## モデル

当面は極小TransformerとQwen3-0.6Bのみを実機検証対象とし、大規模モデル実験を行わない。
Windows用Qwen3-0.6Bは `.tools/models/Qwen3-0.6B` に配置済み。WSLは既存のHugging Faceローカルキャッシュを参照する。モデル重みはGitへ追加しない。
登録パスはOSごとに異なるが、fingerprintは一致する。

## このPCで確認した点

Windows 11 Home / i7-13700K / RAM約96GB / RTX 4080 SUPER 16GB、NVIDIAドライバー616.56。WSL Ubuntu 26.04 LTS。
Windows CPUのスレッド数とCUDA積和精度の調整理由は [数値検証](numerical-validation.md) を参照。
uvの既定保存先で失敗したため、起動・セットアップスクリプトはプロジェクト内のPython/キャッシュ先を明示する。
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
