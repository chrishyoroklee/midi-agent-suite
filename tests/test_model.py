import pytest

from midicore.model import BarMap, Note, Score, TimeSig, parse_pitch, pitch_name


def test_pitch_names_round_trip():
    assert parse_pitch("C4") == 60
    assert parse_pitch("A4") == 69
    assert parse_pitch("Bb3") == 58
    assert parse_pitch("F#5") == 78
    assert parse_pitch("Cb4") == 59
    assert parse_pitch("B#3") == 60
    assert parse_pitch("C-1") == 0
    assert parse_pitch(64) == 64
    assert parse_pitch("64") == 64
    for p in range(128):
        assert parse_pitch(pitch_name(p)) == p


@pytest.mark.parametrize("bad", ["H4", "C", "Cx4", "G10", 128, -1])
def test_parse_pitch_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_pitch(bad)


def test_simple_4_4():
    bm = BarMap([TimeSig(0, 4, 4)])
    assert bm.bar_start(1) == 0
    assert bm.bar_start(5) == 16
    assert bm.position(0) == (1, 1.0)
    assert bm.position(17.5) == (5, 2.5)
    assert bm.beat_of(5, 2.5) == 17.5
    assert bm.bar_of(15.999) == 4
    assert bm.bar_of(16) == 5


def test_pickup_is_bar_zero():
    bm = BarMap([TimeSig(0, 4, 4)], pickup=1.0)
    assert bm.position(0) == (0, 1.0)
    assert bm.bar_of(0.5) == 0
    assert bm.bar_start(1) == 1.0
    assert bm.position(1.0) == (1, 1.0)
    assert bm.position(5.0) == (2, 1.0)
    assert bm.bar_len(0) == 1.0
    assert bm.beat_of(0, 1.0) == 0.0
    with pytest.raises(ValueError):
        bm.beat_of(0, 2.0)  # the pickup is only one beat long


def test_no_bar_zero_without_pickup():
    bm = BarMap([TimeSig(0, 4, 4)])
    with pytest.raises(ValueError):
        bm.bar_start(0)


def test_meter_change():
    # 4 bars of 4/4 then 3/4 from beat 16, then 6/8 from bar 7 (beat 25)
    bm = BarMap([TimeSig(0, 4, 4), TimeSig(16, 3, 4), TimeSig(25, 6, 8)])
    assert bm.bar_start(5) == 16
    assert bm.bar_start(6) == 19
    assert bm.bar_start(7) == 22
    assert bm.bar_start(8) == 25
    assert bm.bar_len(8) == 3.0
    assert bm.position(26.5) == (8, 2.5)
    assert bm.time_sig_of_bar(6).num == 3
    with pytest.raises(ValueError):
        bm.beat_of(6, 4.0)  # 3/4 bar has beats 1..3


def test_meter_change_mid_bar_takes_effect_at_next_barline():
    bm = BarMap([TimeSig(0, 4, 4), TimeSig(2, 3, 4)])
    assert bm.bar_start(2) == 4
    assert bm.bar_len(2) == 3


def test_pickup_with_meter_change():
    bm = BarMap([TimeSig(0, 3, 4), TimeSig(7, 2, 4)], pickup=1.0)
    assert bm.bar_start(1) == 1
    assert bm.bar_start(3) == 7
    assert bm.bar_len(3) == 2
    assert bm.position(9.5) == (4, 1.5)


def test_beat_before_one_rejected():
    with pytest.raises(ValueError):
        BarMap([TimeSig(0, 4, 4)]).beat_of(2, 0.5)


def test_n_bars_note_ending_on_barline():
    s = Score(tracks={"A": [Note(0, 60, 4.0, 80, "A"), Note(28, 60, 4.0, 80, "A")]})
    assert s.n_bars() == 8
    s = Score(tracks={"A": [Note(0, 60, 4.0021, 80, "A")]})
    assert s.n_bars() == 2


def test_n_bars_with_pickup_and_short_last_bar():
    s = Score(tracks={"A": [Note(0, 60, 1.0, 80, "A"), Note(1, 62, 7.0, 80, "A")]}, pickup=1.0)
    assert s.first_bar() == 0
    assert s.n_bars() == 2
