import pytest
import yaml

from video_platform.core.errors import ConfigError
from video_platform.storytelling.choices import localization_overrides, parse_slang, save_local


def test_region_language_slang_mapping():
    o = localization_overrides("Bombay", "hinglish", "low")
    assert o["regional_profile"] == "mumbai" and o["dialect"]["name"] == "bambaiya"
    assert o["dialect"]["intensity"] == 0.25 and o["writing_style"] == "natural_hinglish"
    assert o["character_names"]["region"] == "Mumbai"
    assert localization_overrides("hyderabad")["dialect"]["name"] == "dakhni_urdu"
    assert localization_overrides("neutral")["regional_profile"] == "neutral_hindi"
    assert localization_overrides(slang="0.4") == {"dialect": {"intensity": 0.4, "forced_slang": False}}
    assert parse_slang("none") == 0.0 and parse_slang("high") == 0.8


@pytest.mark.parametrize("bad", [{"region": "gotham"}, {"language": "klingon"}, {"slang": "loads"}, {"slang": "2"}])
def test_bad_inputs_rejected(bad):
    with pytest.raises(ConfigError):
        localization_overrides(**bad)


def test_saved_choice_is_merged_into_config(make_project, tmp_path):
    from video_platform.config.loader import load_project_config

    cfg = make_project()
    save_local(cfg.parent, localization_overrides("delhi", "hindi", "medium"))
    save_local(cfg.parent, localization_overrides(slang="high"))  # later inputs update, earlier ones stay
    loc = load_project_config(cfg).config.localization
    assert loc.regional_profile == "delhi" and loc.writing_style == "natural_hindi" and loc.dialect.intensity == 0.8
    assert yaml.safe_load((cfg.parent / "local.yaml").read_text())["localization"]["dialect"]["name"] == "dilli_hindi"
