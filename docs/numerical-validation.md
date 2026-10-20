# 数値検証

## 自動検証

Windowsネイティブ、Python 3.13.15、PyTorch 2.14.1+cu130で14件のテストに合格。
独立NumPy参照はLayerNorm、Q/K/V、因果Mask、Softmax、集約、出力射影、SwiGLU、Residual、最終Logitsを計算する。
生成位置の対応、PCA固定、変位の原点、共通接頭辞、ZIP往復、ハッシュ、保持・削除・中断時の復旧を検証する。

## 許容誤差

- float32: atol=2e-5, rtol=2e-4（実行時の内部再構成）。
- float16/bfloat16: atol=0.05, rtol=0.04。チェックごとの誤差と判定を保存する。
- 独立NumPy参照テストはatol=1e-6〜2e-6、rtol=2e-5〜3e-5。

保存時はfloat32へ変換する。BF16/FP16の値をfloat32へ拡張するための変換で、
元の推論dtypeはmanifestに別途保存する。モデル重みの量子化は行わない。

## Windows CPU経路

このPCのPyTorchではCPU LayerNormを複数スレッドで実行すると、処理自体は終了するが
faulthandlerがWindowsのネイティブ例外を通知することを再現した。
1スレッドでは通知なく数値検証に合格したため、Windowsの既定CPUスレッド数は1とする。
CUDA演算は別経路。`OBSERVATORY_CPU_THREADS`で明示的に変更できるが、環境ごとの再検証が必要。

## 射影

実験の全Residualを学習対象にする。小規模ではRandomized SVD（Seed固定）、
指定時はExact PCA、大規模ではIncremental PCAを用いる。基底の符号・平均・寄与率を保存する。
比較は両トレースから共通基底を再計算する。Level 1の座標から高次元を逆復元しない。

## 未確定事項

長文・多数軌跡の描画性能、Qwen3-0.6B実重みのGPU誤差、WSL側の独立環境での結果を
実測後に追記する。観測した短文の結果を任意サイズへ一般化しない。
