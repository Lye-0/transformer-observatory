# 数値・実機検証

検証日: 2026-10-20。短文の実測結果を長文・別モデルへ一般化しない。

## 自動検証

WindowsネイティブとWSLの独立cloneでCPU・CUDAを検証する。GitHub ActionsでもWindows/UbuntuのCPU検証とUIビルドを実施する。
独立NumPy参照はLayerNorm、Q/K/V、因果Mask、Softmax、Head集約、出力射影、SwiGLU、全Residual、最終Logitsを計算する。
小型ランダム構成のQwen3/GPT-2でモデル別アダプターも照合する。
生成位置、固定PCA、厳密なDisplacement原点、ゼロベクトルの未定義指標、挿入・削除の対応表、共通接頭辞、取得範囲、容量中断、保持・削除、ZIP往復、OS間モデル再対応、二重起動防止を検証する。
研究機能の自動テストでは、介入の独立参照・自己Patch・強度0・先頭因果性・解除後復元、実験単位の集計、全点t-SNE、処理中の元データ保護、研究ジョブの中止・復旧、入力ハッシュ不一致の拒否、Lensの形式・変換・不一致拒否を検証する。

実行コマンド:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests -q
backend\.venv\Scripts\python.exe -m ruff check --config backend/pyproject.toml backend scripts/validate_model.py scripts/validate_research.py scripts/validate_tuned_lens.py
mise.exe exec -- pnpm --dir frontend run build
mise.exe exec -- pnpm --dir frontend run format:check
```

WSLはPythonを `backend/.venv/bin/python`、miseを `mise` とする。

## 許容誤差と積和精度

- float32内部再構成: atol=2e-5, rtol=2e-4。
- float16/bfloat16: atol=0.05, rtol=0.04。
- 独立NumPy参照: atol=1e-6〜2e-6, rtol=2e-5〜3e-5。
- `max_scaled_error = max(abs(actual-reference)/(atol+rtol*abs(reference)))`。1以下で合格。

BF16/FP16の入力・出力丸めを残差加算、RoPE、Softmax後、SwiGLU積でも再現する。保存値はfloat32へ拡張し、元dtypeを別記する。
CUDAの既定BF16縮小精度累積ではQwen3の最終MLPがCPU参照から許容範囲を超えてずれた。GPUの部分和をfloat32で累積する設定に固定し、FP16縮小精度累積・TF32も無効にした。設定はmanifestに保存する。閾値を緩めて合格させていない。

このPCのWindows CPU LayerNormは複数スレッド時に処理中のネイティブ例外通知を再現した。1スレッドでは通知なく照合に合格するため、Windows既定は1、Linux既定は4スレッドとする。`OBSERVATORY_CPU_THREADS`で変更可能。

## Qwen3-0.6Bの両OS GPU実測

i7-13700K / RAM約96GB / RTX 4080 SUPER 16GB、Python 3.13.15、PyTorch 2.14.1+cu130、Transformers 5.18.0。
同一safetensors・Tokenizer、BF16、Eager Attention、Level 4、全28層・全Head、Attention行列あり。
入力は `The sky is blue.` と `The sky is green.`、それぞれGreedyで2トークン生成。全7位置を実際に処理し、各57観測点を保存した。

| 条件 | 1入力目 | 2入力目 |
|---|---:|---:|
| Windows: 読込〜保存完了 | 88.487秒 | 85.766秒 |
| WSL: 読込〜保存完了 | 51.344秒 | 47.649秒 |
| 内部再構成チェック（両OS） | 756/756合格 | 756/756合格 |
| 最大scaled error（両OS） | 0.15244 | 0.46183 |
| PyTorch CUDA割当ピーク | 約1.12 GiB | 約1.12 GiB |
| Windows RSSサンプル最大 | 約2.25 GiB | 約2.95 GiB |
| WSL RSSサンプル最大 | 約2.14 GiB | 約2.92 GiB |
| トレース容量 | 約38.67 MiB | 約38.67 MiB |

時間は純粋な生成速度ではなく、モデル読込・CPU参照照合・1,939チャンクの同期書込・PCAを含む。RSSは0.1秒間隔の観測で瞬間的ピークを取り逃がし得る。CUDA値はPyTorch割当量で、ドライバー・他プロセスを含む総VRAMではない。
両プロンプトの共通先頭3トークンは、全観測点で高次元差0。重み・Tokenizer fingerprintは両OSで一致した。
生レポート: [Windows](validation/windows-qwen3-0.6b.json)、[WSL](validation/wsl-qwen3-0.6b.json)。計測コミットは各JSONに記載する。

再現コマンド（モデルは自動ダウンロードしない）:

```powershell
backend\.venv\Scripts\python.exe scripts/validate_model.py --model .tools/models/Qwen3-0.6B --output .observatory/validation/local --device cuda
```

WSLでは `--model` にLinux側モデルの絶対パスを渡す。出力はGit管理しない。

## ブラウザとOS間の検証

Windows Chromeで実操作した項目:

- モデル登録、トークナイズ、解析、Scientific/Displacement、Layer・Decode選択。
- Attention行列・Head接続、MLP、Tensor、Pin、Logit Lens、比較Overlay/Side by Side/Difference。
- 解析中のブラウザ再読込と再接続、中止と部分データ確認。
- WSLで作成したZIPをWindowsへインポート。1,939チャンクのSHA-256が一致し、モデル再実行なしで7本・57観測点を描画。
- インポート後、Windows側の同一fingerprintモデルを使うLogit Lensも成功。
- 1,024トークンの極小Transformer: 全1,024本を描画。実際のWebGL線描画で16,384頂点（8区間×2頂点×1,024本）を確認し、約60 FPSを観測。
- 画面幅390 / 768 / 1024 / 1440pxでページと主要操作の横はみ出しなし。コンソール警告・エラーなし。

FPSはこのPC・当該データでの観測で、Qwenの長文や任意トレースの保証ではない。軽量表示は装飾と描画頻度を変え、軌跡を削減しない。

## 研究機能の実モデル検証

コミット`b4660bd`、Qwen3-0.6B、同一の2入力・保存済みトレースを使い、Windows/WSLのそれぞれで10条件を検証した。BF16/CUDA、元入力の末尾位置4で読み出す。Patchの向きはblueを介入先、greenを取得元とする。

| 条件 | Logits差のL2（両OS） | 解除後の最大誤差 |
|---|---:|---:|
| 自己Patch（Block 0後） | 0 | 0 |
| 別入力のPatch（Block 0後） | 25.59224 | 0 |
| 別入力のPatch（最終Block後） | 238.96046 | 0 |
| Layer 0 / Head 0無効化 | 33.18082 | 0 |
| Layer 0 / MLP無効化 | 113.24923 | 0 |
| Head介入の強さ0 | 0 | 0 |
| 末尾のHead無効化、読出位置0 | 0 | 0 |

さらに2実験の集計を独立NumPyと照合した。全399点のt-SNEを同一Seedで2回計算し、各OS内で全座標が完全一致した。元manifestのハッシュは全処理の前後で変化しなかった。OS間の射影座標一致を契約にはしない。
要約: [Windows](validation/windows-research.json)、[WSL](validation/wsl-research.json)。

```powershell
backend\.venv\Scripts\python.exe scripts/validate_research.py --trace-run .observatory/validation/local --output .observatory/validation/research.json
```

`--trace-run`は`validate_model.py`で確定した停止中の検証カタログに限定する。上の10条件には学習済みTuned Lensの予測品質は含めない。このPCに対象重みがなく、Lensは未学習fixtureによる数値検証までである。

Windows ChromeではResearchの3介入、統計、全399点t-SNE、観測点・Token選択、条件付きJSON/CSV、研究履歴の再読込を確認した。各レポートの実行条件を表示し、編集中のフォームと区別する。390 / 768 / 1024 / 1440pxで横はみ出しを確認し、コンソール警告・エラーなし。起動時の前回トレース復元がナビゲーションを上書きする競合も修正した。

## 公開GPT-2 Small・学習済みTuned Lens

Qwen3-0.6B用Lensはダウンロード可能な公開重みを確認できなかったため、GPT-2 Small（124M）の公式公開Lensを使って実学習済み重みでの読出しを検証した。Lens configが指定するモデルrevisionへファイルを合わせ、6ファイルのGit blob/LFS SHA-256とモデルfingerprintを両OSで照合した。[固定URLと照合証跡](validation/gpt2-lens-provenance.json)、[調査の出典と読み出し条件](research.md)を参照。

まずBF16/CUDA、Level 4、全12層、各2トークン生成で内部取得を検証した。両入力とも7位置・25観測点・288チェックに合格し、共通先頭3トークンの高次元差は0だった。

| 条件 | 1入力目 | 2入力目 |
|---|---:|---:|
| Windows: 読込〜保存完了 | 19.529秒 | 16.239秒 |
| WSL: 読込〜保存完了 | 9.721秒 | 7.892秒 |
| 最大scaled error（両OS） | 0.181819 | 0.187266 |
| CUDA割当ピーク | 約0.247 GiB | 約0.247 GiB |
| 保存容量 | 約8.56 MiB | 約8.56 MiB |

生レポート: [Windows GPT-2](validation/windows-gpt2.json)、[WSL GPT-2](validation/wsl-gpt2.json)。時間・容量はQwen3と同じくCPU参照・保存・PCAを含む。BF16許容誤差内のチェックであり、全成分の絶対誤差が小さいことを意味しない。

次にFP32/CUDAで公式学習済みLensを使用し、独自英文3本・各12読み出し点・全位置を評価した。各prompt/読み出し点の平均KLを等しい重みで平均した値である。

| 項目 | Windows | WSL |
|---|---:|---:|
| KL(最終分布 \|\| Logit Lens) | 7.356064 | 7.356064 |
| KL(最終分布 \|\| Tuned Lens) | 1.371441 | 1.371441 |
| APIと直接計算の候補確率の最大差 | 1.602e-7 | 1.602e-7 |

Embeddingから最終Block直前までの12点を含み、最終Norm適用済みhidden_stateを二重にNormへ渡していない。APIはBlock 5出力・入力末尾で照合した。これは短い動作確認であり、汎用コーパスの品質測定ではない。
生レポート: [Windows Lens](validation/windows-gpt2-lens.json)、[WSL Lens](validation/wsl-gpt2-lens.json)。

```powershell
backend\.venv\Scripts\python.exe scripts/validate_model.py --model .tools/models/gpt2 --output .observatory/validation/gpt2 --device cuda
backend\.venv\Scripts\python.exe scripts/validate_tuned_lens.py --model .tools/models/gpt2 --lens .tools/lenses/gpt2 --output .observatory/validation/gpt2-lens --device cuda
```

Windows Chromeでは、保持済みGPT-2サンプルのPredictionでLogit/Tuned Lensを切り替えて計算した。異なるモデルの実験が混在するとき、Patch取得元の候補と統計の選択対象もfingerprintで制限する。
