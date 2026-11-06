"""multiprocessing Pool + 초기화 오류를 호출자에게 전달.

Pool은 initializer에서 난 예외를 호출자에게 안 돌려주고 죽은 작업자를 끝없이 다시 만든다(명령이 안 끝남).
여기서는 초기화 오류를 작업자 안에 잡아 두었다가 작업을 받으면 RuntimeError로 내보낸다 → pool.map이 오류로 끝나고 명령이 실패 코드로 끝남.
정상일 때는 Pool 그대로: 같은 작업·같은 순서·같은 chunksize. 재생 결과에 영향 없음.
"""
import traceback
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool

_INIT_ERROR = None


def _safe_init(init, args):
    global _INIT_ERROR
    try:
        init(*args)
    except (Exception, SystemExit) as e:
        _INIT_ERROR = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _call(job, item):
    if _INIT_ERROR is not None:
        raise RuntimeError(f"작업자 초기화 실패: {_INIT_ERROR}")
    return job(item)


class _Guarded:
    def __init__(self, pool):
        self.pool = pool

    def map(self, job, items, chunksize=None):
        return self.pool.map(partial(_call, job), items, chunksize)


@contextmanager
def guarded_pool(procs, init, args=()):
    """with guarded_pool(10, _init, (…)) as pool: pool.map(job, 목록)"""
    with Pool(procs, _safe_init, (init, args)) as pool:
        yield _Guarded(pool)


def run_pool(procs, init, args, job, items, chunksize=None):
    with guarded_pool(procs, init, args) as pool:
        return pool.map(job, items, chunksize)
