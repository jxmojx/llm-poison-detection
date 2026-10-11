from modelscope.hub.snapshot_download import snapshot_download

# 模型名称（与原代码完全一致，不可修改）
model_name = "damo/nlp_corom_sentence-embedding_chinese-base"
# 下载缓存目录（与原代码一致，确保后续代码能找到模型）
cache_dir = "./model_cache"

# 执行下载（忽略不必要的文件，加速下载）
model_dir = snapshot_download(
    model_name,
    cache_dir=cache_dir,
    ignore_file_pattern=[".git", ".md"]
)

# 打印下载后的本地目录（复制此目录，后续可用于手动验证）
print(f"模型下载完成，本地目录：{model_dir}")