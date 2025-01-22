import ssl
from unittest import mock

from aioredlock.sentinel import Sentinel
from aioredlock.sentinel import SentinelConfigError

import pytest

pytestmark = [pytest.mark.asyncio]


@pytest.fixture
def mocked_redis_sentinel(mocker):
    mock_sentinel = mocker.patch(
        "aioredlock.sentinel.RedisSentinel",
        mock.Mock(
            return_value=mock.AsyncMock(master_for=mock.AsyncMock(return_value=True))
        ),
    )
    return mock_sentinel


@pytest.mark.parametrize(
    "connection,kwargs,expected_kwargs,expected_master,with_ssl",
    (
        (
            {"host": "127.0.0.1", "port": 26379, "master": "leader"},
            {},
            {"sentinels": [("127.0.0.1", 26379)], "max_connections": 100},
            "leader",
            {},
        ),
        (
            "redis://:password@localhost:12345/0?master=whatever&encoding=utf-8",
            {},
            {
                "sentinels": [("localhost", 12345)],
                "db": 0,
                "encoding": "utf-8",
                "password": "password",
                "max_connections": 100,
            },
            "whatever",
            {},
        ),
        (
            "redis://:password@localhost:12345/0?master=whatever&encoding=utf-8",
            {"master": "everything", "password": "newpass", "db": 3},
            {
                "sentinels": [("localhost", 12345)],
                "db": 3,
                "encoding": "utf-8",
                "password": "newpass",
                "max_connections": 100,
            },
            "everything",
            {},
        ),
        (
            "rediss://:password@localhost:12345/2?master=whatever&encoding=utf-8",
            {},
            {
                "sentinels": [("localhost", 12345)],
                "db": 2,
                "encoding": "utf-8",
                "password": "password",
                "max_connections": 100,
            },
            "whatever",
            {"verify_mode": ssl.CERT_REQUIRED, "check_hostname": True},
        ),
        (
            "rediss://:password@localhost:12345/2?master=whatever&encoding=utf-8&ssl_cert_reqs=CERT_NONE",
            {},
            {
                "sentinels": [("localhost", 12345)],
                "db": 2,
                "encoding": "utf-8",
                "password": "password",
                "max_connections": 100,
            },
            "whatever",
            {"verify_mode": ssl.CERT_NONE, "check_hostname": False},
        ),
        (
            "rediss://localhost:12345/2?master=whatever&encoding=utf-8&ssl_cert_reqs=CERT_OPTIONAL",
            {},
            {
                "sentinels": [("localhost", 12345)],
                "db": 2,
                "encoding": "utf-8",
                "password": None,
                "max_connections": 100,
            },
            "whatever",
            {"verify_mode": ssl.CERT_OPTIONAL, "check_hostname": True},
        ),
        (
            ("127.0.0.1", 1234),
            {"master": "blah", "ssl_context": True},
            {
                "sentinels": [("127.0.0.1", 1234)],
                "max_connections": 100,
            },
            "blah",
            {},
        ),
        (
            [("127.0.0.1", 1234), ("blah", 4829)],
            {"master": "blah", "ssl_context": False},
            {
                "sentinels": [("127.0.0.1", 1234), ("blah", 4829)],
                "max_connections": 100,
                "ssl": False,
            },
            "blah",
            {},
        ),
    ),
)
async def test_sentinel(
    ssl_context,
    connection,
    kwargs,
    expected_kwargs,
    expected_master,
    with_ssl,
    mocked_redis_sentinel,
):
    sentinel = Sentinel(connection, **kwargs)
    result = await sentinel.get_master()
    assert result is True
    assert mocked_redis_sentinel.called
    if with_ssl or kwargs.get("ssl_context") is True:
        expected_kwargs["ssl"] = ssl_context
    mocked_redis_sentinel.assert_called_with(**expected_kwargs)
    result = mocked_redis_sentinel.return_value
    assert result.master_for.called
    result.master_for.assert_called_with(expected_master)
    if with_ssl:
        assert ssl_context.check_hostname is with_ssl["check_hostname"]
        assert ssl_context.verify_mode is with_ssl["verify_mode"]


@pytest.mark.parametrize(
    "connection",
    (
        "redis://localhost:1234/0",
        "redis://localhost:1234/blah",
        object(),
    ),
)
async def test_sentinel_config_errors(connection):
    with pytest.raises(SentinelConfigError):
        Sentinel(connection)
