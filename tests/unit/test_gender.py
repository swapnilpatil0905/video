"""Grammatical gender agreement in adapted Hindi dialogue."""

import pytest

from video_platform.storytelling.gender import agreement_issues, norm_gender, short

CASES = [
    ("मैं घर जा रहा हूँ।", "female", None, 1),
    ("मैं घर जा रही हूँ।", "female", None, 0),
    ("मैं कल ज़रूर आऊँगा", "female", None, 1),
    ("मैं कल ज़रूर आऊँगी", "female", None, 0),
    ("मैं वहाँ था।", "female", None, 1),
    ("मैंने उसे देखा था।", "female", None, 0),          # ergative: agrees with the object
    ("मैं जानती थी वो वहाँ था", "female", None, 0),       # था belongs to वो
    ("मैं थक गई हूँ", "male", None, 1),
    ("मैं आदमी हूँ", "male", None, 0),                     # nouns are not guessed
    ("तुम कहाँ जा रहे हो?", "male", "female", 1),
    ("तुम कहाँ जा रही हो?", "male", "female", 0),
    ("तुम जानते हो वो जा रही है", "male", "male", 0),      # third person inside: ambiguous, skipped
    ("आप कब आएँगे?", "male", "female", 1),
    ("आप कब आएंगी?", "male", "male", 1),
    ("क्या तुम मेरे साथ चलोगी?", "male", "male", 1),
    ("तुम कब आओगे?", "male", "female", 1),
    ("तू कब आएगी?", "male", "male", 1),
    ("तू कब आएगा", "male", "male", 0),
    ("मैं ठीक हूँ, तुम बताओ", "female", "male", 0),
    ("हम जा रहे हैं", "female", "female", 0),
    ("आप बैठिए", "male", "female", 0),
]


@pytest.mark.parametrize("text,speaker,listener,expected", CASES)
def test_agreement(text, speaker, listener, expected):
    assert len(agreement_issues(text, speaker, listener)) == expected


def test_unknown_gender_is_not_checked():
    assert agreement_issues("मैं जा रहा हूँ", None, None) == []
    assert norm_gender("Female") == "female" and norm_gender("M") == "male" and norm_gender("unknown") is None
    assert short("woman") == "F" and short(None) == "?"


def test_mentions_other_gender():
    from video_platform.storytelling.gender import mentions_other_gender

    assert mentions_other_gender("A female speaker with a warm voice", "male")
    assert not mentions_other_gender("A male speaker with a warm voice", "male")
    assert not mentions_other_gender("A man and a woman talk", "male")  # both named: not contradictory
    assert not mentions_other_gender("A calm voice", None)


def test_reference_text_is_gender_neutral():
    from video_platform.config.schema import VoiceDesignSpec

    assert agreement_issues(VoiceDesignSpec().reference_text, "female") == []
    assert agreement_issues(VoiceDesignSpec().reference_text, "male") == []
