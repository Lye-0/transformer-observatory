# モデルと取得範囲

更新: 2026-10-20。

| 対象 | 数値検証 | GPU実重み検証 | 制約 |
|---|---|---|---|
| 極小Transformer | 全観測点・最終Logitsを独立NumPyと照合 | Windows/WSL | 固定ランダム重み。事前学習済みLLMではない |
| Qwen3-0.6B | Q/K Norm、RoPE、GQA、Mask、Attention集約・射影、MLP Gate/Up/SiLU/積/Down、残差加算 | Windows/WSL、BF16、全28層、生成あり、各756チェック合格 | 小さい入力で確認。任意長・別サイズまで検証済みとは扱わない |
| GPT-2構造 | Transformersの小型ランダム構成でQKV/GELU/射影・Residualを照合 | 学習済み実重みでは未実測 | 登録可能なアダプター。正式な実モデル検証と区別する |
| 他の構造、MoE、量子化、カスタムコード | 非対応 | 非対応 | 登録時に拒否。Flash Attention内部は対象外 |

ローカルsafetensorsのみを読み、`local_files_only=True`・`trust_remote_code=False`とする。
重み・設定・Tokenizerファイルからfingerprintを作成し、ロード時にも変更がないことを検証する。
登録直後は「取得可能・未検証」。個々の実行で合格した取得範囲・精度・デバイスを記録し、対応能力を更新する。

## 4つの記録レベル

- Level 1: 固定PCA座標・元高次元からの指標・メタデータ。座標確定まではResidualを一時保存し、正常完了時に生値を削除する。
- Level 2: 全Residual観測点を保持。比較・再投影・Logit Lensに必要。
- Level 3/4: 上記に加え、指定Layer・Query Head・テンソル名の取得。空欄はアダプターが提供できる全対象。Level 4も完全な計算グラフの記録ではない。

Attention行列は明示的な選択時のみ保存する。Qwen3/GPT-2ではEager演算のため、保存しなくても推論時のAttention作業領域は必要。
選択範囲にかかわらず、主軌跡のResidualは全トークン・全層で保持する。

## 取得経路と空間

`hook`: 実際のモジュール入出力。`reconstructed`: 実際の取得値・重み・RoPEパラメーター等から再構成し、モデルのAttention出力と照合した値。`backend_mask`: 適用Maskを0/1へ変換したもの。
QwenのRoPE後Q/KとHead別W_O寄与は再構成であり、直接Hook値と区別して表示する。
Q/K/V、MLP中間、Residualは空間名を分ける。Head寄与をResidualに戻す際は実W_Oの対応ブロックを使う。GQAのQuery→KV対応もmanifestに記録する。

## 生成と予測

Qwen3/GPT-2のDecodeはKV Cacheを使い、新しい位置だけ保存する。極小モデルは全コンテキストを再計算するが、履歴の保存は重複させない。
最終生成トークン・EOSも実際にforwardしてから軌跡を付ける。予測時点と処理時点を別々に保存する。
Logit Lensは最終Norm/LM Headによる途中表現の読み出し。途中層が実際に出した確率や、因果的な証明として扱わない。

仕様の拡張機能であるTuned Lens、Activation Patching、Ablation、UMAP/t-SNEは今回の初期リリースには含めない。学習済みLensや介入結果を偽装しない。
