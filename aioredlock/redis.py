import asyncio
import logging
import re
import time
import functools
from itertools import groupby
from typing import Optional
from redis.asyncio import ConnectionPool, Redis as AIORedis

from redis.exceptions import RedisError, ResponseError

from aioredlock.errors import LockError, LockAcquiringError, LockRuntimeError
from aioredlock.sentinel import Sentinel
from aioredlock.utility import clean_password


def all_equal(iterable):
    """Returns True if all the elements are equal to each other"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def raise_error(results, default_message):
    errors = [e for e in results if isinstance(e, BaseException)]
    if any(type(e) is LockRuntimeError for e in errors):
        raise [e for e in errors if type(e) is LockRuntimeError][0]
    elif any(type(e) is LockAcquiringError for e in errors):
        raise [e for e in errors if type(e) is LockAcquiringError][0]
    else:
        raise LockError(default_message) from errors[0]


class Instance:
    # KEYS[1] - lock resource key
    # ARGS[1] - lock unique identifier
    # ARGS[2] - expiration time in milliseconds
    SET_LOCK_SCRIPT = """
    local identifier = redis.call('get', KEYS[1])
    if not identifier or identifier == ARGV[1] then
        return redis.call("set", KEYS[1], ARGV[1], 'PX', ARGV[2])
    else
        return redis.error_reply('ERROR')
    end"""

    # KEYS[1] - lock resource key
    # ARGS[1] - lock unique identifier
    UNSET_LOCK_SCRIPT = """
    local identifier = redis.call('get', KEYS[1])
    if not identifier then
        return redis.status_reply('OK')
    elseif identifier == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return redis.error_reply('ERROR')
    end"""

    # KEYS[1] - lock resource key
    GET_LOCK_TTL_SCRIPT = """
    local identifier = redis.call('get', KEYS[1])
    if not identifier then
        return redis.error_reply('ERROR')
    elseif identifier == ARGV[1] then
        return redis.call("TTL", KEYS[1])
    else
        return redis.error_reply('ERROR')
    end"""

    @staticmethod
    def ensure_connection(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Ensure connection is established before proceeding
            if self._client is None:
                await self.connect()
            # Call the original function
            return await func(self, *args, **kwargs)

        return wrapper

    def __init__(self, connection):
        """
        Redis instance constructor
        Constructor takes single argument - a redis host address
        The address can be one of the following:
         * a dict - {
                        'host': 'localhost',
                        'port': 6379,
                        'db': 0,
                        'password': 'pass'
                    }
            in this case redis.asyncio.Redis will be used;
         * a Redis URI - "redis://host:6379/0?encoding=utf-8";
         * or a unix domain socket path string - "unix://path/to/redis.sock".
         * a redis connection pool.
        :param connection: redis host address (dict, tuple or str)
        """

        self.connection = connection

        self._client: Optional[AIORedis] = None
        self._lock = asyncio.Lock()

        self.set_lock_script_sha1 = None
        self.unset_lock_script_sha1 = None
        self.get_lock_ttl_script_sha1 = None

    @property
    def log(self):
        return logging.getLogger(__name__)

    def __repr__(self):
        connection_details = clean_password(self.connection)
        return "<%s(connection='%s'>" % (self.__class__.__name__, connection_details)

    @staticmethod
    async def _create_redis(*args, **kwargs) -> AIORedis:
        if args[0] is None:
            return AIORedis(**kwargs)

        return AIORedis.from_url(*args, **kwargs)

    async def _register_scripts(self):
        tasks = []
        for script in [
            self.SET_LOCK_SCRIPT,
            self.UNSET_LOCK_SCRIPT,
            self.GET_LOCK_TTL_SCRIPT,
        ]:
            script = re.sub(r"^\s+", "", script, flags=re.M).strip()
            tasks.append(self._client.script_load(script))
        results = await asyncio.gather(*tasks)
        (
            self.set_lock_script_sha1,
            self.unset_lock_script_sha1,
            self.get_lock_ttl_script_sha1,
        ) = (r if isinstance(r, str) else r.decode("utf-8") for r in results)

    async def connect(self) -> AIORedis:
        """
        Get a connection for the self instance
        """
        address, redis_kwargs = None, {}

        if isinstance(self.connection, Sentinel):
            self._client = self.connection.get_master()
        elif isinstance(self.connection, dict):
            assert "host" in self.connection, "Host is not specified"
            redis_kwargs = self.connection
        elif isinstance(self.connection, ConnectionPool):
            conn_kwargs = self.connection.connection_kwargs
            url = f"redis://{conn_kwargs['host']}:{conn_kwargs['port']}/{conn_kwargs['db']}?encoding=utf-8"
            self._client = AIORedis.from_url(url)
        elif isinstance(self.connection, tuple):
            # a tuple ('localhost', 6379, 0, 'pass'), db and password are optional
            redis_kwargs = {
                "host": self.connection[0],
                "port": self.connection[1],
                "db": self.connection[2] if len(self.connection) > 2 else 0,
                "password": self.connection[3] if len(self.connection) > 3 else None,
            }
        else:
            # a string "redis://host:6379/0?encoding=utf-8" or
            # a unix domain socket path "unix:///path/to/redis.sock"
            address = self.connection

        if self._client is None:
            redis_kwargs["max_connections"] = redis_kwargs.get("max_connections", 100)
            async with self._lock:
                if self._client is None:
                    self.log.debug("Connecting %s", repr(self))
                    self._client = await self._create_redis(address, **redis_kwargs)

        if self.set_lock_script_sha1 is None or self.unset_lock_script_sha1 is None:
            await self._register_scripts()

        return self._client

    async def aclose(self):
        """
        Closes connection and resets pool
        """
        if self._client is not None and not isinstance(self.connection, AIORedis):
            try:
                await self._client.aclose()
            except AttributeError:
                await self._client.close()

        self._client = None

    @ensure_connection
    async def set_lock(
        self, resource, lock_identifier, lock_timeout, register_scripts=False
    ):
        """
        Lock this instance and set lock expiration time to lock_timeout
        :param resource: redis key to set
        :param lock_identifier: uniquie id of lock
        :param lock_timeout: timeout for lock in seconds
        :raises: LockError if lock is not acquired
        """

        lock_timeout_ms = int(lock_timeout * 1000)

        try:
            if register_scripts is True:
                await self._register_scripts()
            await self._client.evalsha(
                self.set_lock_script_sha1, 1, resource, lock_identifier, lock_timeout_ms
            )
        except ResponseError as exc:  # script fault
            if exc.args[0].startswith("NOSCRIPT"):
                return await self.set_lock(
                    resource, lock_identifier, lock_timeout, register_scripts=True
                )
            self.log.debug('Can not set lock "%s" on %s', resource, repr(self))
            raise LockAcquiringError("Can not set lock") from exc
        except (RedisError, OSError) as exc:
            self.log.error(
                'Can not set lock "%s" on %s: %s', resource, repr(self), repr(exc)
            )
            raise LockRuntimeError("Can not set lock") from exc
        except asyncio.CancelledError:
            self.log.debug('Lock "%s" is cancelled on %s', resource, repr(self))
            raise
        except Exception:
            self.log.exception('Can not set lock "%s" on %s', resource, repr(self))
            raise
        else:
            self.log.debug('Lock "%s" is set on %s', resource, repr(self))

    @ensure_connection
    async def get_lock_ttl(self, resource, lock_identifier, register_scripts=False):
        """
        Fetch this instance and set lock expiration time to lock_timeout
        :param resource: redis key to get
        :param lock_identifier: unique id of the lock to get
        :param register_scripts: register redis, usually already done, so 'False'.
        :raises: LockError if lock is not available
        """
        try:
            if register_scripts is True:
                await self._register_scripts()

            ttl = await self._client.evalsha(
                self.get_lock_ttl_script_sha1, 1, resource, lock_identifier
            )
        except ResponseError as exc:  # script fault
            if exc.args[0].startswith("NOSCRIPT"):
                return await self.get_lock_ttl(
                    resource, lock_identifier, register_scripts=True
                )
            self.log.debug('Can not get lock "%s" on %s', resource, repr(self))
            raise LockAcquiringError("Can not get lock") from exc
        except (RedisError, OSError) as exc:
            self.log.error(
                'Can not get lock "%s" on %s: %s', resource, repr(self), repr(exc)
            )
            raise LockRuntimeError("Can not get lock") from exc
        except asyncio.CancelledError:
            self.log.debug('Lock "%s" is cancelled on %s', resource, repr(self))
            raise
        except Exception:
            self.log.exception('Can not get lock "%s" on %s', resource, repr(self))
            raise
        else:
            self.log.debug('Lock "%s" with TTL %s is on %s', resource, ttl, repr(self))
            return ttl

    @ensure_connection
    async def unset_lock(self, resource, lock_identifier, register_scripts=False):
        """
        Unlock this instance
        :param resource: redis key to set
        :param lock_identifier: uniquie id of lock
        :raises: LockError if the lock resource acquired with different lock_identifier
        """
        try:
            if register_scripts is True:
                await self._register_scripts()
            await self._client.evalsha(
                self.unset_lock_script_sha1, 1, resource, lock_identifier
            )
        except ResponseError as exc:  # script fault
            if exc.args[0].startswith("NOSCRIPT"):
                return await self.unset_lock(
                    resource, lock_identifier, register_scripts=True
                )
            self.log.debug('Can not unset lock "%s" on %s', resource, repr(self))
            raise LockAcquiringError("Can not unset lock") from exc
        except (RedisError, OSError) as exc:
            self.log.error(
                'Can not unset lock "%s" on %s: %s', resource, repr(self), repr(exc)
            )
            raise LockRuntimeError("Can not unset lock") from exc
        except asyncio.CancelledError:
            self.log.debug('Lock "%s" unset is cancelled on %s', resource, repr(self))
            raise
        except Exception:
            self.log.exception('Can not unset lock "%s" on %s', resource, repr(self))
            raise
        else:
            self.log.debug('Lock "%s" is unset on %s', resource, repr(self))

    @ensure_connection
    async def is_locked(self, resource):
        """
        Checks if the resource is locked by any redlock instance.

        :param resource: The resource string name to check
        :returns: True if locked else False
        """

        lock_identifier = await self._client.get(resource)
        if lock_identifier:
            return True
        else:
            return False


class Redis:
    def __init__(self, redis_connections):
        self.instances = []
        for connection in redis_connections:
            self.instances.append(Instance(connection))

    @property
    def log(self):
        return logging.getLogger(__name__)

    async def set_lock(self, resource, lock_identifier, lock_timeout=10.0):
        """
        Tries to set the lock to all the redis instances

        :param resource: The resource string name to lock
        :param lock_identifier: The id of the lock. A unique string
        :param lock_timeout: lock's lifetime
        :return float: The elapsed time that took to lock the instances
            in seconds
        :raises: LockRuntimeError or LockAcquiringError or LockError if the lock has not
            been set to at least (N/2 + 1) instances
        """
        start_time = time.monotonic()

        successes = await asyncio.gather(
            *[
                i.set_lock(resource, lock_identifier, lock_timeout)
                for i in self.instances
            ],
            return_exceptions=True,
        )
        successful_sets = sum(s is None for s in successes)

        elapsed_time = time.monotonic() - start_time
        locked = successful_sets >= int(len(self.instances) / 2) + 1

        self.log.debug(
            'Lock "%s" is set on %d/%d instances in %s seconds',
            resource,
            successful_sets,
            len(self.instances),
            elapsed_time,
        )

        if not locked:
            raise_error(successes, 'Can not acquire the lock "%s"' % resource)

        return elapsed_time

    async def get_lock_ttl(self, resource, lock_identifier=None):
        """
        Tries to get the lock from all the redis instances

        :param resource: The resource string name to fetch
        :param lock_identifier: The id of the lock. A unique string
        :return float: The TTL of that lock reported by redis
        :raises: LockRuntimeError or LockAcquiringError or LockError if the lock has not
            been set to at least (N/2 + 1) instances
        """
        start_time = time.monotonic()
        successes = await asyncio.gather(
            *[i.get_lock_ttl(resource, lock_identifier) for i in self.instances],
            return_exceptions=True,
        )
        successful_list = [s for s in successes if not isinstance(s, BaseException)]
        # should check if all the value are approx. the same with math.isclose...
        locked = len(successful_list) >= int(len(self.instances) / 2) + 1
        success = all_equal(successful_list) and locked
        elapsed_time = time.monotonic() - start_time

        self.log.debug(
            'Lock "%s" is set on %d/%d instances in %s seconds',
            resource,
            len(successful_list),
            len(self.instances),
            elapsed_time,
        )

        if not success:
            raise_error(successes, 'Could not fetch the TTL for lock "%s"' % resource)

        return successful_list[0]

    async def unset_lock(self, resource, lock_identifier):
        """
        Tries to unset the lock to all the redis instances

        :param resource: The resource string name to lock
        :param lock_identifier: The id of the lock. A unique string
        :return float: The elapsed time that took to lock the instances in iseconds
        :raises: LockRuntimeError or LockAcquiringError or LockError if the lock has no
            matching identifier in more then (N/2 - 1) instances
        """

        if not self.instances:
            return 0.0

        start_time = time.monotonic()

        successes = await asyncio.gather(
            *[i.unset_lock(resource, lock_identifier) for i in self.instances],
            return_exceptions=True,
        )
        successful_removes = sum(s is None for s in successes)

        elapsed_time = time.monotonic() - start_time
        unlocked = successful_removes >= int(len(self.instances) / 2) + 1

        self.log.debug(
            'Lock "%s" is unset on %d/%d instances in %s seconds',
            resource,
            successful_removes,
            len(self.instances),
            elapsed_time,
        )

        if not unlocked:
            raise_error(successes, "Can not release the lock")

        return elapsed_time

    async def is_locked(self, resource):
        """
        Checks if the resource is locked by any redlock instance.

        :param resource: The resource string name to lock
        :returns: True if locked else False
        """

        successes = await asyncio.gather(
            *[i.is_locked(resource) for i in self.instances], return_exceptions=True
        )
        successful_sets = sum(s is True for s in successes)

        return successful_sets >= int(len(self.instances) / 2) + 1

    async def clear_connections(self):
        self.log.debug("Clearing connection")

        if self.instances:
            coros = []
            while self.instances:
                coros.append(self.instances.pop().aclose())
            await asyncio.gather(*coros)
