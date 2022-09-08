from siatune.version import parse_version_info


def test_version_check():
    assert parse_version_info('2.24.1rc') > parse_version_info('2.24.1')
    assert parse_version_info('2.24.1') > parse_version_info('2.24.0rc')
