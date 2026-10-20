# 研究機能

Researchでは観察済みトレースを入力に、介入・記述統計・追加射影を別のレポートとして保存する。元のResidual・固定PCA・指標は変更しない。数値として得られた出力差と、その解釈を区別する。

## 介入実験

同じ重み・Tokenizerの登録をfingerprintで照合し、元入力のPrefillを再実行する。基準、介入あり、Hook解除後の3回を同じモデルインスタンスで計算する。解除後が基準へ戻らなければ失敗とする。

- Residual Patch: 保存済みのEmbeddingまたはBlock出力（`layer.N.mlp`）を指定位置へ代入する。取得元のSHA-256も照合する。Attention後のResidual Patchには対応しない。
- Head Ablation: W_O直前の連結Head出力の指定Query Head領域を縮小する。GQAではQuery Headの番号を使う。
- MLP Ablation: Residualへ加算するMLP出力を縮小する。

強さ1は完全置換・無効化、0は介入なしの対照、途中は線形混合・縮小。位置は0始まりで、空欄は入力の末尾。Patch取得元と介入先の位置数は揃える。取得元には保存済み生成位置も明示指定できるが、介入先・出力読出位置は元入力Prefillに限定する。生成系列は再生成しない。

読出位置での次トークン分布についてKL（基準→介入後）、Total Variation、Logits差L2、両分布の上位10候補の和集合と任意Token IDの確率差を記録する。解除後のLogits最大誤差も記録する。対象位置・強さ・入力・dtypeに依存する結果で、自然な因果機構や一般的な重要性まで保証しない。CPUへフォールバックした場合はその環境をレポートへ明記する。

## 複数実験の記述統計

同じモデル・観測点・指標定義の完了実験を2〜20件選択する。有効なトークンの指標を各実験内で平均し、その実験平均を等しい重みで集計する。実験間の標準偏差はddof=1。未定義値は除外し、有効トークン数・実験数を残す。同じ実験の重複選択を拒否する。

独立PCAに依存する`projected_update`は除外する。各トークンを独立した実験反復とは扱わず、p値・統計的有意差・因果性は出さない。異なる入力長や生成条件は記述統計の解釈に影響する。

## 全点t-SNE

Level 2以上の全トークン×全観測点を3次元へ射影する。近傍の探索用表示であり、線の長さや点間距離を元高次元の距離・変位として解釈しない。Scientific/Displacementの固定PCAは維持する。Seed・Perplexity・反復数・KLと全座標を保存する。

全4,096観測点、かつ計算容量の概算が設定RAM予算以内のときだけ実行する。上限超過時は抽出せずエラーにする。Barnes-Hut、PCA初期値、learning_rate=auto、最大750反復を使用する。中止はfit区間の終了時に反映する。OS/ライブラリ間の同一座標は保証せず、同じ実行環境での再現性を検証する。

## Tuned Lens

[AlignmentResearchの公式実装](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py)の`linear_tuned_lens`に対応する。変換は`h + affine(h)`で、Embeddingはtranslator 0、Block N出力はtranslator N+1。最終Block出力にはtranslatorを使わず、最終Norm/LM Headで読む。Attention後には適用しない。

[公式のローカル読込手順](https://github.com/AlignmentResearch/tuned-lens/blob/main/docs/source/tutorials/loading_pretrained_lenses.rst)にある`config.json`と`params.pt`、または同じstate_dictを保存した`params.safetensors`を含むフォルダーをResearchから登録する。重みのキーは`0.weight`・`0.bias`等の層別連番。次元・層数・bias設定・有限値・全キーを検証する。

このアプリではモデルの取り違えを避けるため、configへ`observatory_model_fingerprint`を追加する。値はResearchの対象モデル欄に表示される。**実際にその重み・Tokenizerを対象に学習したことを確認してから**指定する。名前や次元が同じという理由だけで関連付けない。学習情報は任意の`training_info`に記載できる。

TorchのZIP形式は`weights_only=True`・CPU mmapで読み、古い非ZIP形式は拒否する。safetensorsを優先する。512MiB・設定RAM予算の上限、登録時からのチェックサム変更も検証する。登録後はExplorerのPredictionでLensを選択する。結果はLensによる補正後の読み出しであり、途中層の実出力や正しい解釈の証明とは扱わない。

このPCの既存`llm-lab`では学習済みLensが見つからなかった。通常のQwen3重みは別物である。実装の読込・変換・Stage対応は明示的な未学習の数値fixtureでテストした。Qwen3の学習済みLensによる予測品質は未検証。実モデルでその検証を行うには、対象モデルに対応する学習済みLens、またはLens学習用のデータ・学習条件が必要になる。

## 保存と再現

研究処理は推論・Lens読出しと直列に実行する。処理中の元トレースは削除・保持先への移動・自動整理から保護する。確定レポートはカタログのSQLiteへ保存し、ブラウザを閉じてもバックエンド処理は続く。再起動時は中断ジョブを失敗へ移し、確定レポートを維持する。

画面とJSONは対象実験・実行条件を示す。JSONにはfingerprint、入力manifestのハッシュ、元のdtype/device・整合性情報、実行環境、結果と復元誤差を含む。CSVは候補確率・集計表・全射影座標の結果表で、条件の保管にはJSONも併用する。元トレースのZIPに研究レポートは含めない。レポート自体が元トレースを永続保持するわけではないため、再解析したい実験はLibraryで「保持」を設定する。

`scripts/validate_model.py`の完了・停止済み検証出力に対し、実モデルの介入対照・統計・t-SNEの10条件を再現できる。

```powershell
backend\.venv\Scripts\python.exe scripts/validate_research.py --trace-run .observatory/validation/local --output .observatory/validation/research.json
```

アプリが使用中のカタログへこのスクリプトを接続しない。WSLではLinux側の専用cloneと`backend/.venv/bin/python`を使う。
