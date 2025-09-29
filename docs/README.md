# Documentation Index

本リポジトリのドキュメント一覧です。目的に応じて適切なガイドをご参照ください。

## 🖥️ OS別推奨環境

機械学習研究では**Linux環境**が最も安定して動作します。以下の推奨に従うことで、トラブルを最小限に抑えることができます。

### Windows ユーザー
**🥇 強く推奨**: **WSL2 + Ubuntu 22.04**
- ✅ Linux環境での最大互換性
- ✅ CUDA-WSL対応でGPU活用可能
- ✅ 開発効率とトラブル解決が容易
- 📖 詳細: [WSL2セットアップ](installation-os-specific.md#wsl2-setup)

**🥈 代替案**: ネイティブWindows
- ⚠️ 一部ライブラリの互換性問題あり
- 📖 詳細: [Windows直接インストール](installation-os-specific.md#windows)

### macOS ユーザー
**Intel Mac**: ネイティブ環境で問題なし
**Apple Silicon (M1/M2/M3)**: Metal Performance Shaders (MPS) 対応
- ✅ ネイティブ環境推奨
- ⚠️ 一部ARM64未対応ライブラリに注意
- 📖 詳細: [macOS環境構築](installation-os-specific.md#macos)

### Linux ユーザー
**🥇 最適**: Ubuntu 20.04+
- ✅ 全機能フル対応、推奨継続

---

## 🚀 新規参加者向け（推奨読書順）

### 1. [GitHub基本ガイド](github-guide.md)
**対象**: GitHub初心者の学生
**内容**: 本プロジェクトでGitHubを使った協働開発を学ぶための完全ガイド
- なぜGitHubを学ぶのか（研究における価値）
- VS CodeのGitHub拡張機能を使った実践的操作
- Issue → Branch → Pull Requestのワークフロー
- AI支援ツール（Claude Code、Cline等）の活用方法
- トラブルシューティングと質問方法

### 2. [システム要件](requirements.md)
**対象**: 環境構築前の確認用
**内容**: 実験実行に必要なハードウェア・ソフトウェア要件
- GPU/CPU/メモリの最小・推奨スペック
- モデルサイズ別VRAM要件表
- ネットワーク・ストレージ考慮事項
- パフォーマンス期待値とトラブルシューティング

## ⚙️ 技術セットアップ

### 3. [環境セットアップガイド](setup-guide.md)
**対象**: 開発環境を構築する全員
**内容**: Python環境とML開発環境の詳細セットアップ手順
- 仮想環境（venv/conda）の作成と管理
- GPU/CUDA/PyTorchのインストール
- HuggingFace Hub認証設定
- 設定最適化とメモリ管理
- 検証とトラブルシューティング

### 4. [OS別インストール手順](installation-os-specific.md)
**対象**: 特定OS向けの詳細手順が必要な方
**内容**: オペレーティングシステム別の具体的インストール方法
- **Linux** (Ubuntu/Debian): APTパッケージ管理、CUDA設定
- **Windows** 10/11: Visual Studio Build Tools、PowerShell設定
- **macOS**: Homebrew、Apple Silicon (M1/M2) 対応
- Docker環境での統一セットアップ
- OS固有の問題と解決方法

## 📊 研究リソース

### 5. [LLMモデル候補](model-candidates.md)
**対象**: モデル選定・評価を担当する学生
**内容**: 色生成実験に適したLocal LLMの最新候補リスト（2024-2025）
- **推奨モデル**: Phi-4-mini、Qwen2.5、Llama-3.3等
- Function calling対応モデルの重点紹介
- ハードウェア別推奨モデル（8GB/16GB/24GB+ GPU）
- 学生向け評価タスクと調査方法論
- 性能 vs リソース要件の比較表

---

## 📋 全ドキュメント一覧

| ドキュメント | サイズ | 最終更新 | 主な対象者 |
|-------------|--------|----------|-----------|
| [github-guide.md](github-guide.md) | 14KB | 最新 | GitHub初心者 |
| [setup-guide.md](setup-guide.md) | 6KB | Phase1 | 環境構築者 |
| [requirements.md](requirements.md) | 3KB | Phase1 | システム管理者 |
| [installation-os-specific.md](installation-os-specific.md) | 8KB | Phase1 | OS別セットアップ |
| [model-candidates.md](model-candidates.md) | 9KB | 最新 | モデル評価担当 |

## 🎯 読書ガイド

### 初回セットアップの場合
1. **GitHub基本ガイド** → **システム要件** → **環境セットアップガイド** → **OS別インストール手順**

### モデル研究の場合
1. **システム要件** → **LLMモデル候補** → 実際のモデルテスト

### トラブル時
1. **該当ドキュメント**のトラブルシューティングセクション → **GitHub Issues**で質問

---


## 📝 ドキュメント更新

ドキュメントの改善提案や間違いを見つけた場合は、Pull Requestまたは Issue でお知らせください。