# GitHub基本ガイド - Local LLM Analysis Project

このガイドでは、本リポジトリで研究活動を行うために必要なGitHubの基本操作を、実践的に説明します。

## なぜGitHubを学ぶのか？

プロジェクトのゴールや現在実施中のタスクの共有、そして今後取り組むべき課題などをチームで共有しながら進めるために、GitHubの活用は有用です。GitHubの活用方法に慣れておくことは、今後、他のプロジェクトにおいても役立つことが多くなります。

## 🎯 このガイドの目的

- リポジトリをクローンして開発環境をセットアップする
- Issue（課題）をベースとした開発の流れを理解する
- VS CodeのGitHub拡張機能を使った効率的な作業方法を身につける
- Pull Requestを作成してコード共有・レビューを行う

## ✅ 前提条件

- GitHubアカウントを持っている
- 本リポジトリへの招待メールを受信済み
- VS Codeがインストール済み

---

## 1. 📧 招待を受け入れる

### リポジトリアクセス権の確認

1. **招待メールを確認**
   - 件名: "You've been invited to join the toorpia-labs/local-llm-analysis repository"
   - メール内の **"View invitation"** ボタンをクリック

2. **招待を受諾**
   - GitHubページで **"Accept invitation"** をクリック
   - リポジトリページにリダイレクトされることを確認

3. **アクセス確認**
   - https://github.com/toorpia-labs/local-llm-analysis にアクセス
   - リポジトリの内容が見えることを確認

---

## 2. 🔧 VS Code環境の準備

### 必要な拡張機能のインストール

VS Codeを開いて以下の拡張機能をインストールしてください：

#### 必須拡張機能
1. **GitHub Pull Requests and Issues** (`GitHub.vscode-pull-request-github`)
   - **最重要**: Issues、PR管理がVS Code内で完結
   - インストール: 拡張機能パネル → "GitHub Pull Requests" で検索

2. **Git Graph** (`mhutchie.git-graph`)
   - コミット履歴を視覚的に確認できる
   - インストール: 拡張機能パネル → "Git Graph" で検索

#### 推奨拡張機能
- **Python** (`ms-python.python`) - Python開発用

### Gitの初期設定

**ターミナル**（VS Code内でも可）で以下を実行：

```bash
# 自分の名前を設定（GitHubアカウント名でOK）
git config --global user.name "あなたの名前"

# 自分のメールアドレスを設定（GitHubアカウントのメール）
git config --global user.email "your.email@example.com"

# 設定確認
git config --global --list
```

---

## 2.5. 🤖 AI支援ツールの活用（強く推奨！）

### GitHubで困ったらAIに聞こう

Git/GitHubの操作は最初は複雑に感じるかもしれませんが、**AI支援ツール**を使えば安心です！

#### おすすめのVS Code拡張機能

以下のAI支援ツールから **どれか1つ** をインストールすることを強く推奨します：

1. **Claude Code** - このガイドを作成したClaude AIのVS Code拡張
2. **Cline** - 高機能なAI開発支援ツール
3. **Cursor** - AI統合開発環境
4. **GitHub Copilot** - GitHub公式AI支援

### AI支援ツールでできること

#### Git/GitHub操作の質問例
```
「Issue #1から作業ブランチを作る方法を教えて」
「コミットメッセージの書き方がわからない」
「Pull Request の作り方を教えて」
「マージコンフリクトが起きた時の解決方法は？」
「VS CodeでGitHubのIssueを確認する方法は？」
```

#### Python実装の質問例
```
「TransformersLoaderクラスの基本構造を教えて」
「PyTorchでモデルを読み込む方法は？」
「エラーメッセージの意味がわからない」
「テストコードの書き方を教えて」
```

### 🎯 AI活用の基本的な流れ

1. **困ったことがあったら → まずAIに質問**
2. **エラーが出たら → エラーメッセージをAIに見せる**
3. **コードの書き方がわからない → 具体例を聞く**
4. **それでも解決しない → IssueコメントやDiscordで相談**

### 💡 効果的な質問のコツ

#### ❌ 曖昧な質問
```
「gitがわからない」
「エラーが出る」
```

#### ✅ 具体的な質問
```
「VS CodeでIssue #1に取り組むためのブランチ作成方法を教えて」
「以下のエラーメッセージの解決方法は？
[エラーメッセージをコピペ]」
```

### 重要なポイント

- **GitHub操作で挫折する必要はありません**
- **わからないことは全てAIが教えてくれます**
- **失敗しても大丈夫 - gitなら元に戻せます**
- **最初はみんな初心者です**

AI支援ツールがあれば、このガイドの内容も **「わからなくなったらその都度AIに聞く」** というスタンスで進められます！

---

## 3. 📥 リポジトリをクローンする

### VS Codeでクローンする

1. **VS Codeを開く**
2. **Command Palette**を開く
   - `Ctrl+Shift+P` (Windows/Linux) または `Cmd+Shift+P` (Mac)
3. **Git: Clone**と入力・選択
4. **リポジトリURL**を入力：
   ```
   https://github.com/toorpia-labs/local-llm-analysis
   ```
5. **保存先フォルダ**を選択（例：`/home/username/projects/`）
6. **"Open"**をクリックしてプロジェクトを開く

### GitHub拡張機能の認証

初回使用時、GitHubアカウントでの認証が必要です：

1. **左側縦バーのGitHubアイコン**（👤のような形）をクリック
2. **"Sign in to GitHub"** をクリック
3. ブラウザでGitHubログインを完了
4. VS Codeに戻ると認証完了

---

## 4. 📋 Issue Drivenな開発とは

### Issueの役割

**Issue**（課題）は以下の役割を果たします：
- **タスク管理**: 何をやるべきかを明確化
- **進捗共有**: チーム全体で作業状況を把握
- **議論の場**: 技術的な相談や質問
- **履歴管理**: 変更理由や経緯を記録

### VS CodeでIssueを確認する方法

1. **左側縦バーのGitHubアイコン**をクリック
2. **Issues**セクションを展開
3. 現在のIssueが一覧表示されます：
   - **Issue #1**: Implement TransformersLoader module
   - **Issue #2**: Implement MCPController for tool calling
   - **Issue #3**: Implement Analyzer module

### 担当Issueの選択

1. **自分の技術レベル**や**興味**に基づいて担当Issueを選択
2. **Issue #1**が最も基礎的で取り組みやすい
3. **複数人で同じIssue**に取り組んでも構いません

---

## 5. 🔄 実践：Issue #1に取り組む手順

ここでは**Issue #1**を例に、VS CodeのGitHub拡張機能を使った実際のワークフローを説明します。

### Step 1: Issueの詳細確認

1. **GitHubパネル** → **Issues** → **Issue #1**をクリック
2. Issue の詳細内容を確認
3. **不明点があれば**Issue コメント欄で質問

### Step 2: Issueから直接作業開始（🔥重要）

これがVS Code GitHub拡張の最大の利点です！

1. **Issues一覧**で**Issue #1**にマウスオーバー
2. **右向き矢印（→）**が表示される
3. **矢印をクリック**

すると自動的に：
- 適切な名前のブランチが作成される（例：`1-implement-transformersloader-module`）
- 作業用ブランチに切り替わる
- すぐに実装作業を開始できる状態になる

### Step 3: 実装作業

1. **該当ファイル**を編集：
   ```
   experiments/color_generation/src/model_loaders/__init__.py
   ```

2. **進捗に応じて小まめにコミット**：
   - ファイル保存後、**ソース管理パネル**（左縦バー）をクリック
   - 変更ファイルの **"+"** をクリック（ステージング）
   - **コミットメッセージ**を入力：
     ```
     Add TransformersLoader class skeleton

     - Define basic class structure
     - Add placeholder methods for model loading
     - Addresses #1
     ```
   - **✓ コミット**をクリック

### Step 4: リモートにプッシュ

1. **ソース管理パネル**で **"Publish Branch"** または **"Push"** をクリック
2. 自動的にリモートリポジトリにブランチがプッシュされる

### Step 5: Pull Request作成

1. **GitHubパネル** → **Pull Requests** → **"Create Pull Request"**
2. または、プッシュ後に表示される通知から **"Create Pull Request"**

3. **PR情報**を入力：
   ```
   タイトル: Implement TransformersLoader module

   説明:
   ## 概要
   Issue #1の解決のため、TransformersLoaderモジュールを実装しました。

   ## 実装内容
   - [x] TransformersLoaderクラスの基本構造
   - [x] モデル読み込み機能のスケルトン
   - [ ] 隠れ状態抽出機能（次回PR予定）

   ## テスト
   - [x] 基本的なインポートテスト
   - [ ] 実際のモデル読み込みテスト（実装後）

   Closes #1
   ```

4. **Create Pull Request**

---

## 6. 🔍 Pull Requestのお作法

### 良いPRタイトルの例

```
✅ 良い例:
- "Implement TransformersLoader module"
- "Add RGB color generation tool for MCP experiment"
- "Fix memory leak in hidden state extraction"

❌ 避けるべき例:
- "update"
- "fix bug"
- "作業中"
```

### PR説明に含めるべき内容

1. **概要**: 何を実装・修正したか
2. **変更内容**: 具体的な変更点のリスト
3. **テスト**: 動作確認方法
4. **関連Issue**: `Closes #1` で自動クローズ

### レビューの流れ

1. **VS Code GitHubパネル**でPRの状態を確認
2. **コメントやレビュー**がついたら通知が表示される
3. **修正が必要**な場合：
   - 同じブランチでファイルを修正
   - コミット・プッシュすると自動的にPRに反映

---

## 7. 🚨 困ったときのヘルプ

### よくあるエラーと対処法

#### 認証エラー
```
remote: Support for password authentication was removed...
```
**解決法**: VS CodeのGitHub拡張で再認証、またはPersonal Access Tokenを使用

#### ブランチ作成に失敗
- **原因**: ローカルブランチとリモートブランチの同期問題
- **解決法**:
  1. **ソース管理パネル** → **...** → **Pull**
  2. 最新状態に更新してから再度Issue矢印をクリック

#### Issues が表示されない
- **原因**: GitHubアカウント認証の問題
- **解決法**:
  1. **GitHubパネル**で**"Sign in to GitHub"**
  2. ブラウザで認証を完了

### VS Code GitHub拡張機能のポイント

#### Issues パネルの活用
- **担当者でフィルタ**: 自分の担当Issueのみ表示
- **状態でフィルタ**: Open/Closedを切り替え
- **右クリックメニュー**: Issue作成・編集・コメント

#### Pull Requests パネルの活用
- **Review Changes**: エディタ内でコードレビュー
- **Comment**: 行単位でコメント追加
- **Approve/Request Changes**: レビュー結果の送信

### 質問・相談方法

1. **Issue コメント**: 技術的な質問はIssue内で（VS Codeから直接可能）
2. **PR レビュー**: コードに関する質問はPRで
3. **Discord/Slack**: 緊急時や一般的な相談
4. **対面**: 複雑な問題の場合

---

## 8. 🎯 効率的なワークフロー（推奨手順）

### 日常的な作業の流れ

1. **VS Code起動**
2. **GitHubパネル**で新しいIssueや通知を確認
3. **担当Issue**にマウスオーバー → **矢印クリック**でブランチ作成・切り替え
4. **実装作業**（エディタ）
5. **コミット**（ソース管理パネル）
6. **プッシュ**（ソース管理パネル）
7. **PR作成**（GitHubパネル）
8. **レビュー待ち・対応**（GitHubパネル）

### 複数Issue並行作業のコツ

1. **Issue毎にブランチ分離**が自動で行われる
2. **ソース管理パネル**でブランチ切り替え可能
3. **GitHubパネル**で各Issueの進捗を一覧確認

---

## 9. 🎉 次のステップ

1. **Issue #1, #2, #3**から自分の担当を選択
2. **Issue矢印クリック** → **実装** → **PR作成**のサイクルを実践
3. **他の学生のPR**もVS Code内でレビュー
4. **疑問点**はIssueコメントやDiscordで質問

## 📚 参考リンク

- [VS Code GitHub Extension](https://code.visualstudio.com/docs/editor/github)
- [GitHub Pull Requests and Issues 拡張機能](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github)
- [Git基本コマンドチートシート](https://education.github.com/git-cheat-sheet-education.pdf)

---

**Happy Coding! 🚀**

*VS CodeのGitHub拡張機能を使えば、ブラウザを開かずにすべての作業が完結します！*