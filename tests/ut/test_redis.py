from __future__ import annotations
import asyncio
import hashlib
from unittest.mock import call, patch, AsyncMock, Mock, AsyncMockMixin

from redis.asyncio import ConnectionPool
from redis.exceptions import ResponseError
import pytest

from aioredlock.errors import LockError, LockAcquiringError, LockRuntimeError
from aioredlock.redis import Instance, Redis
from aioredlock.sentinel import Sentinel


def calculate_sha1(text):
    sha1 = hashlib.sha1()
    sha1.update(text.encode())
    digest = sha1.hexdigest()
    return digest


EVAL_OK = b"OK"
EVAL_ERROR = ResponseError("ERROR")
CANCELLED = asyncio.CancelledError("CANCELLED")
CONNECT_ERROR = OSError("ERROR")
RANDOM_ERROR = Exception("FAULT")


@pytest.fixture
async def fake_client() -> FakeClient:
    _fake_pool = FakeClient()
    return _fake_pool


class FakeClient(AsyncMockMixin):
    SET_IF_NOT_EXIST = "SET_IF_NOT_EXIST"

    def __init__(self):
        super().__init__()
        self.script_cache = {}
        self.connection_kwargs = {}
        self.evalsha = AsyncMock(return_value=True)
        self.get = AsyncMock(return_value=False)
        self.script_load = AsyncMock(side_effect=self._fake_script_load)
        self.execute_command = AsyncMock(side_effect=self._fake_execute_command)
        self.aclose = AsyncMock(return_value=True)
        self.release = AsyncMock()

    def is_fake(self):
        # Only for development purposes
        return True

    def _fake_script_load(self, script):
        digest = calculate_sha1(script)
        self.script_cache[digest] = script

        return digest.encode()

    def _fake_execute_command(self, *args):
        cmd = b" ".join(args[:2])
        if cmd == b"SCRIPT LOAD":
            return self._fake_script_load(args[-1])

    def _fake_execute(self, *args):
        cmd = b" ".join(args[:2])
        if cmd == b"SCRIPT LOAD":
            return self._fake_script_load(args[-1])


def fake_create_redis(fake_client):
    """
    Original Redis pool have magick method __await__ to create exclusive
    connection. MagicMock sees this method and thinks that Redis pool
    instance is awaitable and tries to await it.
    To avoit this behavior we are using this constructor with Mock.side_effect
    instead of Mock.return_value.
    """

    async def create_redis(*args, **kwargs):
        return fake_client

    return create_redis


class TestInstance:
    script_names = ["SET_LOCK_SCRIPT", "UNSET_LOCK_SCRIPT", "GET_LOCK_TTL_SCRIPT"]

    def test_initialization(self):
        instance = Instance(("localhost", 6379))

        assert instance.connection == ("localhost", 6379)
        assert instance._client is None
        assert isinstance(instance._lock, asyncio.Lock)

        # scripts
        for name in self.script_names:
            assert getattr(instance, "%s_sha1" % name.lower()) is None

    @pytest.mark.parametrize(
        "connection, expected_address, expected_kwargs",
        [
            (
                ("localhost", 6379),
                None,
                {"host": "localhost", "port": 6379, "db": 0, "password": None},
            ),
            (
                {"host": "localhost", "port": 6379, "db": 0, "password": "pass"},
                None,
                {"host": "localhost", "port": 6379, "db": 0, "password": "pass"},
            ),
            (
                "redis://host:6379/0?encoding=utf-8",
                "redis://host:6379/0?encoding=utf-8",
                {},
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_connect_pool_not_created(
        self, connection, expected_address, expected_kwargs, fake_client
    ):
        with patch(
            "aioredlock.redis.Instance._create_redis",
            AsyncMock(return_value=fake_client),
        ) as create_redis_pool:
            instance = Instance(connection)

            assert instance._client is None
            pool = await instance.connect()

            create_redis_pool.assert_called_once_with(
                expected_address, **expected_kwargs, max_connections=100
            )
            assert pool is fake_client
            assert instance._client is fake_client

            # scripts
            assert pool.script_load.call_count == len(self.script_names)
            for name in self.script_names:
                digest = getattr(instance, "%s_sha1" % name.lower())
                assert digest
                assert digest in pool.script_cache
            await fake_client.aclose()

    @pytest.mark.asyncio
    async def test_connect_pool_not_created_with_max_connections(self, fake_client):
        connection = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "pass",
            "max_connections": 5,
        }
        with patch(
            "aioredlock.redis.Instance._create_redis",
            AsyncMock(return_value=fake_client),
        ) as create_redis_pool:
            instance = Instance(connection)

            assert instance._client is None
            pool = await instance.connect()

            create_redis_pool.assert_called_once_with(None, **connection)
            assert pool is fake_client
            assert instance._client is fake_client

    async def test_connect_pool_already_created(self, fake_client):
        with patch(
            "aioredlock.redis.Instance._create_redis",
            AsyncMock(return_value=fake_client),
        ) as create_redis_pool:
            instance = Instance(("localhost", 6379))
            fake_client = FakeClient()
            instance._client = fake_client
            pool = await instance.connect()

            assert not create_redis_pool.called
            assert pool is fake_client
            assert pool.script_load.called is True

    @pytest.mark.asyncio
    async def test_connect_pool_aioredis_instance(self, mocker, fake_client):
        pool = AsyncMock(spec=ConnectionPool)
        pool.connection_kwargs = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "secret",
        }
        mocker.patch("redis.asyncio.ConnectionPool", return_value=pool)

        mocker.patch("redis.asyncio.Redis.from_url", return_value=fake_client)
        instance = Instance(pool)

        assert instance._client is None
        await instance.connect()
        assert fake_client.script_load.call_count == len(self.script_names)
        assert instance.set_lock_script_sha1 is not None
        assert instance.unset_lock_script_sha1 is not None

    @pytest.mark.asyncio
    async def test_connect_pool_aioredis_instance_with_sentinel(self, fake_client):
        sentinel = Sentinel(("127.0.0.1", 26379), master="leader")
        with patch("redis.asyncio.Sentinel.master_for", Mock(return_value=fake_client)):
            instance = Instance(sentinel)

            assert instance._client is None
            await instance.connect()

        assert fake_client.script_load.call_count == len(self.script_names)
        assert instance.set_lock_script_sha1 is not None
        assert instance.unset_lock_script_sha1 is not None

    @pytest.fixture
    def fake_instance(self, fake_client):
        with patch(
            "aioredlock.redis.Instance._create_redis",
            AsyncMock(return_value=fake_client),
        ) as create_redis:
            create_redis.side_effect = fake_create_redis(fake_client)
            instance = Instance(("localhost", 6379))
            yield instance

    @pytest.mark.asyncio
    async def test_lock(self, fake_instance: Instance):
        instance = fake_instance
        await instance.connect()
        redis_client = instance._client

        await instance.set_lock("resource", "lock_id", 10.0)

        redis_client.evalsha.assert_called_once_with(
            instance.set_lock_script_sha1, 1, "resource", "lock_id", 10000
        )

    @pytest.mark.asyncio
    async def test_get_lock_ttl(self, fake_instance: Instance):
        instance = fake_instance
        await instance.connect()
        redis_client = instance._client

        await instance.get_lock_ttl("resource", "lock_id")
        redis_client.evalsha.assert_called_with(
            instance.get_lock_ttl_script_sha1, 1, "resource", "lock_id"
        )

    @pytest.mark.asyncio
    async def test_lock_sleep(self, fake_instance: Instance):
        loop = asyncio.get_running_loop()
        instance = fake_instance

        async def hold_lock(instance):
            async with instance._lock:
                await asyncio.sleep(0.1)
                instance._client = FakeClient()

        await loop.create_task(hold_lock(instance))
        await asyncio.sleep(0.1)
        await instance.connect()
        redis_client = instance._client

        await instance.set_lock("resource", "lock_id", 10.0)

        redis_client.evalsha.assert_called_once_with(
            instance.set_lock_script_sha1, 1, "resource", "lock_id", 10000
        )

        instance._client = None
        await instance.aclose()
        assert redis_client.aclose.called is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "func,args,expected_keys,expected_args",
        (
            (
                "set_lock",
                ("resource", "lock_id", 10.0),
                ["resource"],
                ["lock_id", 10000],
            ),
            ("unset_lock", ("resource", "lock_id"), ["resource"], ["lock_id"]),
            ("get_lock_ttl", ("resource", "lock_id"), ["resource"], ["lock_id"]),
        ),
    )
    async def test_lock_without_scripts(
        self, fake_instance: Instance, func, args, expected_keys, expected_args
    ):
        instance = fake_instance
        await instance.connect()
        redis_client = instance._client
        redis_client.evalsha.side_effect = [
            ResponseError("NOSCRIPT"),
            AsyncMock(return_value=True),
        ]

        await getattr(instance, func)(*args)

        assert redis_client.evalsha.call_count == 2
        assert redis_client.script_load.call_count == 6  # for 3 scripts.

        redis_client.evalsha.assert_called_with(
            getattr(instance, "{0}_script_sha1".format(func)),
            1,
            *expected_keys,
            *expected_args,
        )

    @pytest.mark.asyncio
    async def test_unset_lock(self, fake_instance: Instance):
        instance = fake_instance
        await instance.connect()
        redis_client = instance._client

        await instance.unset_lock("resource", "lock_id")

        redis_client.evalsha.assert_called_once_with(
            instance.unset_lock_script_sha1, 1, "resource", "lock_id"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_return_value,locked",
        [
            (b"lock_identifier", True),
            (None, False),
        ],
    )
    async def test_is_locked(self, fake_instance: Instance, get_return_value, locked):
        instance = fake_instance
        await instance.connect()
        redis_client = instance._client

        redis_client.get = AsyncMock(return_value=get_return_value)

        res = await instance.is_locked("resource")

        assert res == locked
        redis_client.get.assert_called_once_with("resource")


@pytest.fixture
def redis_two_connections():
    return [{"host": "localhost", "port": 6379}, {"host": "127.0.0.1", "port": 6378}]


@pytest.fixture
def redis_three_connections():
    return [
        {"host": "localhost", "port": 6379},
        {"host": "127.0.0.1", "port": 6378},
        {"host": "8.8.8.8", "port": 6377},
    ]


@pytest.fixture
def mock_redis_two_instances(redis_two_connections, fake_client):
    redis = Redis(redis_two_connections)

    for instance in redis.instances:
        instance._client = fake_client

    return redis


@pytest.fixture
def mock_redis_three_instances(redis_three_connections, fake_client):
    redis = Redis(redis_three_connections)

    for instance in redis.instances:
        instance._client = fake_client

    return redis


class TestRedis:
    def test_initialization(self, redis_two_connections):
        with patch("aioredlock.redis.Instance.__init__") as mock_instance:
            mock_instance.return_value = None

            redis = Redis(redis_two_connections)

            calls = [
                call({"host": "localhost", "port": 6379}),
                call({"host": "127.0.0.1", "port": 6378}),
            ]
            mock_instance.assert_has_calls(calls)
            assert len(redis.instances) == 2

    parametrize_methods = pytest.mark.parametrize(
        "method_name, call_args",
        [
            ("set_lock", (1, "resource", "lock_id", 10000)),
            ("unset_lock", (1, "resource", "lock_id")),
            ("get_lock_ttl", (1, "resource", "lock_id")),
        ],
    )

    @pytest.mark.asyncio
    @parametrize_methods
    async def test_lock(
        self, mock_redis_two_instances, fake_client, method_name, call_args
    ):
        redis = mock_redis_two_instances

        method = getattr(redis, method_name)

        await method("resource", "lock_id")

        script_sha1 = getattr(redis.instances[0], "%s_script_sha1" % method_name)

        calls = [call(script_sha1, *call_args)] * 2
        fake_client.evalsha.assert_has_calls(calls, any_order=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_return_value,locked",
        [
            (b"lock_identifier", True),
            (None, False),
        ],
    )
    async def test_is_locked(
        self, mock_redis_two_instances, fake_client, get_return_value, locked
    ):
        redis = mock_redis_two_instances

        fake_client.get = AsyncMock(return_value=get_return_value)

        res = await redis.is_locked("resource")

        calls = [call("resource")] * 2
        fake_client.get.assert_has_calls(calls)
        assert res == locked

    @pytest.mark.asyncio
    @parametrize_methods
    async def test_lock_one_of_two_instances_failed(
        self, mock_redis_two_instances, fake_client, method_name, call_args
    ):
        redis = mock_redis_two_instances
        fake_client.evalsha = AsyncMock(side_effect=[EVAL_ERROR, EVAL_OK])

        method = getattr(redis, method_name)

        with pytest.raises(LockError):
            await method("resource", "lock_id")

        script_sha1 = getattr(redis.instances[0], "%s_script_sha1" % method_name)

        calls = [call(script_sha1, *call_args)] * 2
        fake_client.evalsha.assert_has_calls(calls)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "redis_result, success",
        [
            ([EVAL_OK, EVAL_OK, EVAL_OK], True),
            ([EVAL_OK, EVAL_OK, EVAL_ERROR], True),
            ([EVAL_OK, EVAL_ERROR, CONNECT_ERROR], False),
            ([EVAL_ERROR, EVAL_ERROR, CONNECT_ERROR], False),
            ([EVAL_ERROR, CONNECT_ERROR, RANDOM_ERROR], False),
            ([CANCELLED, CANCELLED, CANCELLED], False),
        ],
    )
    @parametrize_methods
    async def test_three_instances_combination(
        self,
        fake_client,
        mock_redis_three_instances,
        redis_result,
        success,
        method_name,
        call_args,
    ):
        redis = mock_redis_three_instances
        fake_client.evalsha = AsyncMock(side_effect=redis_result)

        method = getattr(redis, method_name)

        if success:
            await method("resource", "lock_id")
        else:
            with pytest.raises(LockError) as exc_info:
                await method("resource", "lock_id")
            assert hasattr(exc_info.value, "__cause__")
            assert isinstance(exc_info.value.__cause__, BaseException)

        script_sha1 = getattr(redis.instances[0], "%s_script_sha1" % method_name)

        calls = [call(script_sha1, *call_args)] * 3
        fake_client.evalsha.assert_has_calls(calls)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "redis_result, error",
        [
            ([EVAL_OK, EVAL_ERROR, CONNECT_ERROR], LockRuntimeError),
            ([EVAL_ERROR, EVAL_ERROR, CONNECT_ERROR], LockRuntimeError),
            ([EVAL_ERROR, CONNECT_ERROR, RANDOM_ERROR], LockRuntimeError),
            ([EVAL_ERROR, EVAL_ERROR, EVAL_OK], LockAcquiringError),
            ([CANCELLED, CANCELLED, CANCELLED], LockError),
            ([RANDOM_ERROR, CANCELLED, CANCELLED], LockError),
        ],
    )
    @parametrize_methods
    async def test_three_instances_combination_errors(
        self,
        fake_client,
        mock_redis_three_instances,
        redis_result,
        error,
        method_name,
        call_args,
    ):
        redis = mock_redis_three_instances
        fake_client.evalsha = AsyncMock(side_effect=redis_result)

        method = getattr(redis, method_name)

        with pytest.raises(error) as exc_info:
            await method("resource", "lock_id")

        assert hasattr(exc_info.value, "__cause__")
        assert isinstance(exc_info.value.__cause__, BaseException)

        script_sha1 = getattr(redis.instances[0], "%s_script_sha1" % method_name)

        calls = [call(script_sha1, *call_args)] * 3
        fake_client.evalsha.assert_has_calls(calls)

    @pytest.mark.asyncio
    async def test_clear_connections(self, mock_redis_two_instances, fake_client):
        redis = mock_redis_two_instances
        fake_client.aclose = AsyncMock(return_value=True)

        await redis.clear_connections()

        fake_client.aclose.assert_has_calls([call(), call()])
        fake_client.aclose.reset_mock()

        await redis.clear_connections()

        assert fake_client.aclose.called is False

    @pytest.mark.asyncio
    async def test_get_lock(self, mock_redis_two_instances, fake_client):
        redis = mock_redis_two_instances

        await redis.get_lock_ttl("resource", "lock_id")

        script_sha1 = getattr(redis.instances[0], "get_lock_ttl_script_sha1")

        calls = [call(script_sha1, 1, "resource", "lock_id")]
        fake_client.evalsha.assert_has_calls(calls)
