from helpers import CONFIG_PATH, raises
from lsfvg.config import ConfigError, config_hash, load_config


def test_default_config_loads_and_hashes_stably():
    a, b = load_config(CONFIG_PATH), load_config(CONFIG_PATH)
    assert a.active_instruments == ["EUR_USD", "SPX500_USD"]
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(CONFIG_PATH, {"stops": {"stop_buffer_atr": 0.2}}))


def test_unknown_key_and_wrong_type_are_reported_together():
    try:
        load_config(CONFIG_PATH, {"stops": {"bogus": 1, "min_stop_atr": "x"}})
    except ConfigError as e:
        msg = str(e)
        assert "unknown key 'bogus'" in msg and "min_stop_atr" in msg
    else:
        raise AssertionError("expected ConfigError")


def test_cross_field_validation():
    with raises(ConfigError, "min_stop_atr must be <="):
        load_config(CONFIG_PATH, {"stops": {"min_stop_atr": 3.0, "max_stop_atr": 1.0}})
    with raises(ConfigError, "base <= ltf <= mtf < htf"):
        load_config(CONFIG_PATH, {"timeframes": {"ltf": "30m"}})
    with raises(ConfigError, "allow_live"):
        load_config(CONFIG_PATH, {"run": {"mode": "live"}})
    with raises(ConfigError, "needs filters.news.csv_path"):
        load_config(CONFIG_PATH, {"filters": {"news": {"enabled": True}}})
    with raises(ConfigError, "not one of"):
        load_config(CONFIG_PATH, {"stops": {"stop_mode": "magic"}})
    # live is possible only with the explicit flag
    assert load_config(CONFIG_PATH, {"run": {"mode": "live", "allow_live": True}}).run.mode == "live"
