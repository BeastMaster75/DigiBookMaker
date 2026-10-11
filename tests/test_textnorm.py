from digibook.textnorm import clean_ocr_text, normalize_for_search


def test_cjk_spaces_removed():
    assert clean_ocr_text("図書 館 は 朝 早く") == "図書館は朝早く"
    assert clean_ocr_text("Tokyo 東京 is big") == "Tokyo 東京 is big"


def test_arabic_folding():
    assert normalize_for_search("المكتبة") == normalize_for_search("المكتبه")
    assert normalize_for_search("أحمد") == normalize_for_search("احمد")
    assert normalize_for_search("كِتَاب") == normalize_for_search("كتاب")


def test_case_and_width():
    assert normalize_for_search("ＨＥＬＬＯ World") == "hello world"
