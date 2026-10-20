# 開発環境と着手前の決定

更新日: 2026-10-20

## 文書の位置付け

`Transformer_Observatory_Development_Specification.md` は、提供された仕様書 v1.0 の原文コピー。
この文書は、その後のユーザーとの確認で決まった環境条件を記録する。
Python版については原文の3.12系という提案に代わり、以下の3.13.15を採用する。

現段階は環境確認・設定・文書整備まで。アプリの実装、依存ライブラリの導入、モデル重みの配置はまだ行わない。

## 配置と管理

- Windows作業リポジトリ: `C:\Users\kawau\dev\LLM\transformer-observatory`
- Windowsのプロジェクト専用仮想環境: `backend/.venv`
- WSLの既存参考環境: `/home/lye/dev/llm/llm-lab/.venv`
- WSL用の本プロジェクトのクローンは、Linux側ファイルシステムに別途作成する。配置候補: `/home/lye/dev/llm/transformer-observatory`
- Python環境はuv、Node.jsとpnpmはmiseで管理する。仮想環境とnode_modulesはOSごとに作成する。
- 依存ライブラリとWindows側のモデル準備は実装着手時に行う。
- 仮想環境、モデル重み、生成トレース、キャッシュはGit管理対象にしない。着手時に.gitignoreへ反映する。
- 当面の検証対象は数値照合用の極小TransformerとQwen3-0.6B。

## 採用するツールの版

| ツール | 版 | 管理方法・理由 |
|---|---|---|
| Python | 3.13.15 | uv。既存WSL環境とそろえる。Windows側は今後セットアップして検証する。 |
| Node.js | 24.21.0 | mise.toml。Windowsに導入済みのLTS版で、実体の起動を確認済み。 |
| pnpm | 11.24.0 | mise.toml。Windowsに導入済みで、実体の起動を確認済み。 |

Node.jsとpnpmはルートのmise.tomlを両OSで共有する。Pythonはmise側へ重複登録しない。
実装着手時にPython版を.python-version等へ反映し、依存ライブラリの版・ロックファイルを確定する。

## 確認済みの実行環境

- Windows 11 Home、Core i7-13700K、RAM約96GB、RTX 4080 SUPER 16GB。
- NVIDIAドライバー616.56。CドライブはNTFS、調査時の空き約173GiB。
- WSL Ubuntu 26.04 LTS。既存llm-labはPython 3.13.15、torch 2.14.1（CUDA 13.0）、transformers 5.18.0、accelerate 1.15.0。
- 既存のQwen3-0.6Bをローカルファイルのみでロードし、4トークンのGPU forwardと内部状態取得に成功。出力値の有限性を確認。
- 上記は環境の動作確認であり、仕様が要求する各Block内部観測点の数値検証を済ませたことは意味しない。

## セットアップ時の注意

- 通常のWindows Pythonは3.11.2。グローバルPythonを置き換えず、uvで本プロジェクト用3.13.15を用意する。
- 調査時はpnpmの通常呼び出しとmise配下の実体で版が異なった。プロジェクト内ではmise.tomlを適用して実行する。
- 自動実行ではPowerShellプロファイルのmise連携がPSConsoleReadLineエラーを起こすことがある。非対話コマンドではプロファイルを読み込まない実行方法を使う。
- uvの既定キャッシュでは権限エラーが出た。調査用の書き込み可能なキャッシュ先を明示すると解消した。セットアップ時にも保存先を確認する。
- Ubuntu仮想ディスクの実体はCドライブ上にある。容量管理はホスト側の空きも参照する。
- WindowsのGitはcore.autocrlf=true。着手時に.gitattributesでシェルスクリプトなどのLFを明示する。

## ユーザーが行うWSLのmise導入

UbuntuのBashで次を実行する。mise本体の導入作業はユーザーが担当する。

```bash
curl -fsSL https://mise.run | sh
~/.local/bin/mise --version
```

Bashで有効にするため、次の行を `~/.bashrc` の末尾に1回だけ追加する。

```bash
eval "$(~/.local/bin/mise activate bash)"
```

新しいUbuntuターミナルを開き、確認する。

```bash
mise --version
```

mise本体を導入しただけでは、Node.js・pnpmはまだ導入されない。
本プロジェクトをWSLへcloneした後、そのルートで以下を実行すれば共有設定の版を導入・確認できる。
この操作は後の環境準備時に行う。

```bash
mise trust
mise install
mise exec -- node --version
mise exec -- pnpm --version
```

期待値はそれぞれ `v24.21.0` と `11.24.0`。

公式資料:
- https://mise.jdx.dev/installing-mise.html
- https://mise.jdx.dev/getting-started.html
- https://mise.jdx.dev/lang/node.html
