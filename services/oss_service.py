import oss2
from ..config import settings

class OSSService:
    def __init__(self):
        self._bucket = None

    @property
    def bucket(self):
        if self._bucket is None:
            if not settings.OSS_BUCKET_NAME or settings.OSS_BUCKET_NAME == "your_bucket_name":
                raise ValueError("OSS_BUCKET_NAME is not configured in .env")
            auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
            self._bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)
        return self._bucket

    def upload_file(self, file_content, filename, directory="laws", headers=None):
        """
        上传文件到 OSS 指定目录
        :param file_content: 文件内容（bytes）
        :param filename: 文件名
        :param directory: OSS 目录，默认为 "laws"
        :param headers: 可选的 HTTP 头部信息
        :return: OSS key (完整路径，不以 / 开头)
        """
        # 移除目录开头的 /（如果有）
        directory = directory.lstrip('/')
        # 确保目录以 / 结尾（用于构建路径）
        if not directory.endswith('/'):
            directory = directory + '/'
        
        # 构建完整的 OSS key（不以 / 开头）
        oss_key = f"{directory}{filename}"
        self.bucket.put_object(oss_key, file_content, headers=headers)
        return oss_key

    def get_file_url(self, oss_key):
        """
        获取文件的公开访问 URL
        :param oss_key: OSS 存储的 key（完整路径）
        :return: 文件的公开访问 URL
        """
        # 确保 oss_key 不以 / 开头
        if oss_key.startswith('/'):
            oss_key = oss_key[1:]
        return f"https://{settings.OSS_BUCKET_NAME}.{settings.OSS_ENDPOINT}/{oss_key}"

    def delete_file(self, file_url_or_key):
        """
        从 OSS 删除文件
        :param file_url_or_key: 文件的完整访问 URL 或 OSS key
        """
        if not file_url_or_key:
            return
        try:
            if file_url_or_key.startswith('http'):
                # 提取 key。URL 格式: https://bucket.endpoint/key
                from urllib.parse import urlparse
                parsed_url = urlparse(file_url_or_key)
                oss_key = parsed_url.path.lstrip('/')
            else:
                oss_key = file_url_or_key.lstrip('/')
                
            if oss_key:
                self.bucket.delete_object(oss_key)
        except Exception as e:
            print(f"Failed to delete OSS file {file_url_or_key}: {e}")

oss_service = OSSService()
