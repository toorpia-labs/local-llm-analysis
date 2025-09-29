# Documentation Index

本リポジトリのドキュメント一覧です。目的に応じて適切なガイドをご参照ください。


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

## 📊 研究リソース・研究成果

### 5. [LLM信頼性の根本的課題：単純タスクでの出力不安定性の実証研究](llm-reliability-simple-tasks.md) 🔥
**内容**: 本研究プロジェクトの主要な研究成果
- **重要な発見**: 極めて単純な色指定タスクでも10-20%の失敗率
- **実証実験**: 赤色90%・紫色80%の成功率（microsoft/Phi-4-mini-instruct）
- **問題の構造**: 出力制御失敗 + 色認識誤りの多層的課題
- **技術的解決策**: Hidden State解析による確定的出力制御
- **社会的意義**: AI安全性への根本的問題提起と産業応用への警鐘

### 6. [LLMモデル候補](model-candidates.md)
**内容**: 色生成実験に適したLocal LLMの最新候補リスト（2024-2025）
- **推奨モデル**: Phi-4-mini、Qwen2.5、Llama-3.3等
- Function calling対応モデルの重点紹介
- ハードウェア別推奨モデル（8GB/16GB/24GB+ GPU）
- 段階的モデル評価方法論
- 性能 vs リソース要件の比較表

---

## 📋 全ドキュメント一覧

| ドキュメント | サイズ | 最終更新 | 主な対象者 |
|-------------|--------|----------|-----------|
| [**llm-reliability-simple-tasks.md**](llm-reliability-simple-tasks.md) | **11KB** | **🔥最新** | **研究者・学術関係者** |
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