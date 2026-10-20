# トレース形式 v1

実験ごとの32桁UUIDフォルダーに、`manifest.json`、`.trace-owner`、`integrity.json`、数値NPYチャンクを格納する。pickleは使用しない。

## manifest

- モデルの設定・重み・Tokenizer fingerprint、元の登録パス、取得条件、プロンプト、実トークンIDと位置。
- 生成トークンの予測Stepと、実際に入力して内部値を取得したStep。
- OS・Python・PyTorch・Transformers・CUDA・GPU・元dtype・積和精度。
- `embedding`、`layer.N.attention`、`layer.N.mlp`の定義。最終Normは主Residual観測点から除外する。
- 固定PCAの平均・基底・符号規約・寄与率・計算法・Seed・学習行数。
- チャンクごとの相対パス、SHA-256、shape、dtype、kind、stage、layer、step、positions、space、Head番号、provenance。
- 名前・説明・タグ・保持状態・保存先・完了状態・中断理由・各数値チェック。

保存した値はfloat32（Mask等はuint8）。QwenのBF16値をfloat32へ拡張しても、元の情報精度は増えない。
Q/K/Vなどは空間名を分け、再構成値をHookからの直接取得と区別する。

## 配列

| 内容 | shape | 意味 |
|---|---|---|
| Residual | `[取得位置数, hidden]` | 1観測点・1生成Stepの実値 |
| Q/K/V | `[取得位置数, 選択Head数, head_dim]` | Head番号は別メタデータ |
| Attention係数 | `[選択Query Head数, 取得Query位置数, Key位置数]` | Decode時は過去Keyを含む |
| Scientific / Displacement | `[全処理位置数, 観測点数, 3]` | 同じ固定基底。Displacement初期位置は厳密に0 |
| 指標・有効性 | `[全処理位置数, 観測点数, 7]` | 未定義の角度等はAPIでnull／UIで「—」 |

Decodeでは新しい位置だけ保存し、既存位置のResidualを複製しない。最終生成位置も実際にforwardしてから完了する。
Level 1はPCAと指標確定後に生Residualを削除するため、高次元の再解析・共通PCA比較には使えない。

## 書込・容量・復旧

1観測点ずつCPUへ取り出して同期書込する。無制限の非同期キューや全層のGPU履歴を保持しない。
各NPYをflush/fsyncして原子的に置換し、manifestを逐次確定する。ディスク上限と空き容量を各チャンクで監視する。
PCAはRAM予算に応じてRandomized / Incrementalを選び、Exact指定が予算を超えれば拒否する。
推論・CPU参照計算・Eager Attentionの作業領域はPCA予算と別で、Preflightに概算を表示する。OS全体の厳密なRAM上限を設定するものではない。
バックエンド再起動時は終了していないジョブを失敗・不完全とし、確定済み部分チャンクを残す。同じカタログの二重起動はOSロックで拒否する。

## 保持・削除・ポータブルZIP

一時領域と永続カタログはOS既定のTemp/XDGへ分離する。設定画面で新しいトレースの保存先を変えても、既存実験は元の場所から閲覧する。
「保持」は `data_root/retained/traces` へコピーして全ハッシュを照合し、その後に一時コピーを削除する。保持解除後は自動整理の対象になり得る。
一時整理は期限・容量の方針を適用する。保持中・実行中を除外し、手動削除はプレビューと確認を要求する。所有マーカー・専用UUID・リンク/ジャンクションの拒否で親フォルダーの誤削除を防ぐ。

エクスポートはmanifest・整合性情報・NPYのZIP。モデル重みは含めない。
インポートは容量・パス・重複・バージョン・位置/形状・有限値・全ハッシュを検証し、新しいUUIDで永続領域へ保持する。
Windows/WSL間ではZIPを移動する。元OSの絶対パスを表示に使っても、保存先として採用しない。
座標閲覧にモデルは不要。Logit Lensなど再実行を伴う処理では、このOSに登録済みの同一fingerprintモデルを探す。
未知のschema_versionは推測して読まず拒否する。将来の移行処理は別バージョンとして追加する。

## 研究レポート

Researchの確定レポートはSQLiteの`studies`へJSONとして保存する。schema_version=1、要求条件、参照実験、モデルfingerprint、manifestハッシュ、環境と結果を含む。中断した研究ジョブは再起動時に失敗へ移し、確定レポートは維持する。
参照中のトレースは削除・保持先への移動・自動整理から保護する。完了後も再解析したい元データは明示的に「保持」を設定する。研究JSON/CSVはトレースZIPと別にエクスポートする。詳細は[研究機能](research.md)を参照。
