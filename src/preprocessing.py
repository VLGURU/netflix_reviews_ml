import re

PHONE_RE = re.compile(r"(\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
NAME_RE = re.compile(r"\b(Иван|Пётр|Петров|Сидоров|Александр|Мария|Анна|Елена)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def mask_phone(text: str) -> str:
    return PHONE_RE.sub("[ТЕЛЕФОН]", text)


def mask_name(text: str) -> str:
    return NAME_RE.sub("[ИМЯ]", text)


def mask_email(text: str) -> str:
    return EMAIL_RE.sub("[EMAIL]", text)


def mask_pii(text: str) -> str:
    text = mask_phone(text)
    text = mask_email(text)
    text = mask_name(text)
    return text


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    return clean_text(mask_pii(text))