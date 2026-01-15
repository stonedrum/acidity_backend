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

    def upload_file(self, file_content, filename):
        # filename is the key in OSS
        self.bucket.put_object(filename, file_content)
        return filename

    def get_file_url(self, filename):
        return f"https://{settings.OSS_BUCKET_NAME}.{settings.OSS_ENDPOINT}/{filename}"

oss_service = OSSService()
