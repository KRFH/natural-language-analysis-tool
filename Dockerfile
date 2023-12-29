# 基本となるイメージを指定
FROM python:3.9

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のコピーとインストール
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコピー
COPY . /app/

# アプリケーションの起動
CMD ["python", "app.py"]
