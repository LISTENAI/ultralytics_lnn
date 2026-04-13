# Thirdparty Dependencies

本目录用于存放第三方依赖库，需要手动下载。

## 下载方法

### 方法1: 使用 Python 脚本（推荐）
```bash
python download_thirdparty.py
```

### 方法2: 使用 Bash 脚本
```bash
bash download_thirdparty.sh
```

### 方法3: 手动克隆
```bash
# 克隆 linger (量化训练框架)
git clone --depth 1 https://github.com/LISTENAI/linger.git thirdparty/linger

# 克隆 thinker (模型打包框架)
git clone --depth 1 https://github.com/LISTENAI/thinker.git thirdparty/thinker
```

## 安装依赖

```bash
# 安装 linger
cd thirdparty/linger && pip install -e .

# 安装 thinker
cd thirdparty/thinker && pip install -e .
cd thirdparty/thinker/tools && pip install -e .
```

## 相关链接

- Linger: https://github.com/LISTENAI/linger
- Thinker: https://github.com/LISTENAI/thinker