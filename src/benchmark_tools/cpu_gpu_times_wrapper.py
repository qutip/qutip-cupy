import warnings

# from pytest_benchmark.fixture import BenchmarkFixture
from pytest_benchmark.fixture import FixtureAlreadyUsed
from pytest_benchmark.stats import Metadata
from cupyx.time import repeat as cp_repeat


class GpuWrapper(object):
    """
    This class will wrap a  pytest-benchmark  type Fixture
    in order to provide separate CPU and GPU timings.
    The GpuWrapper instance will share all its attributes
    transparently with the pytest-benchmark fixture instance.
    Ths class relies on CuPy's experimental timing
    utility ``cupyx.time.repeat``
    """

    def __init__(self, wrapped_class, iterations=30, rounds=5, warmup_rounds=10):
        # we need to initialize wrapped_class this way
        # since we are overriding set_attr
        self.__dict__["wrapped_class"] = wrapped_class
        self.iterations = iterations
        self.rounds = rounds
        self.warmup_rounds = warmup_rounds

    def __getattr__(self, attr):
        # orig_attr = self.wrapped_class.__getattribute__(attr)
        orig_attr = getattr(self.wrapped_class, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_class:
                    return self
                return result

            return hooked
        else:
            return orig_attr

    def __setattr__(self, attr, value):
        setattr(self.wrapped_class, attr, value)

    def pedanticupy(
        self,
        function_to_benchmark,
        args=(),
        kwargs={},
        iterations=30,
        rounds=5,
        warmup_rounds=10,
    ):
        """
        By using this method you have the same control on the benchmark as when using
        pytest-benchamrk's own ``pedantic``
        with the exception of providing an special setup function.
        """
        if self._mode:
            self.has_error = True
            raise FixtureAlreadyUsed(
                "Fixture can only be used once. Previously it was used in %s mode."
                % self._mode
            )
        try:
            self._mode = "benchmark.pedantic(...)"
            return self._raw_pedantic(
                function_to_benchmark,
                args,
                kwargs,
                iterations=iterations,
                rounds=rounds,
                warmup_rounds=warmup_rounds,
            )
        except Exception:
            self.has_error = True
            raise

    def _raw_pedantic(
        self,
        function_to_benchmark,
        args=(),
        kwargs={},
        iterations=30,
        rounds=5,
        warmup_rounds=10,
    ):

        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Must have positive int for `iterations`.")

        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError("Must have positive int for `rounds`.")

        if not isinstance(warmup_rounds, int) or warmup_rounds < 0:
            raise ValueError("Must have positive int for `warmup_rounds`.")

        iterations = self.iterations
        rounds = self.rounds
        warmup_rounds = self.warmup_rounds

        if self.enabled:
            self.stats = self._make_stats(iterations)
            self.stats.group = "device_all"
            self.statscpu = self._make_stats(iterations)
            self.statscpu.group = "device_cpu"
            self.statsgpu = self._make_stats(iterations)
            self.statsgpu.group = "device_gpu"

            self._logger.debug(
                "  Running %s rounds x %s iterations ..." % (rounds, iterations),
                yellow=True,
                bold=True,
            )

            for _ in range(rounds):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action="ignore",
                        category=FutureWarning,
                        message=r"cupyx.time.repeat is experimental.",
                    )

                    results = cp_repeat(
                        function_to_benchmark,
                        args,
                        kwargs,
                        n_warmup=warmup_rounds,
                        max_duration=self._max_time,
                        n_repeat=iterations,
                    )

                for tim_cpu, tim_gpu in zip(results.cpu_times, results.gpu_times[0]):

                    self.stats.update(tim_cpu + tim_gpu)
                    self.statscpu.update(tim_cpu)
                    self.statsgpu.update(tim_gpu)

        function_result = function_to_benchmark(*args, **kwargs)
        return function_result

    def _make_stats(self, iterations):
        bench_stats = Metadata(
            self,
            iterations=iterations,
            options={
                "disable_gc": self._disable_gc,
                "timer": self._timer,
                "min_rounds": self._min_rounds,
                "max_time": self._max_time,
                "min_time": self._min_time,
                "warmup": self._warmup,
            },
        )
        self._add_stats(bench_stats)

        return bench_stats
