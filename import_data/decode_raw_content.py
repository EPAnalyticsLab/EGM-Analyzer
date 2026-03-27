import base64


def decode_raw_content(content, encoding="utf-8"):
    content_raw = content.split(",")[1]
    decoded = base64.b64decode(content_raw)
    return decoded.decode(encoding)
