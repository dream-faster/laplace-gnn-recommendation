import pstats
import cProfile as profile


class Profiler:
    def __init__(self, every: int = 100):
        self.prof = profile.Profile()
        self.prof.enable()

        self.every = every

    def print_stats(self, i: int):
        if i % self.every == 0:
            stats = pstats.Stats(self.prof).strip_dirs()
            stats.dump_stats("stats.dmp")
            for j, aspect in enumerate(["tottime"]):
                if j % 2 == 0:
                    print("\x1b[1;37;40m")
                else:
                    print("\x1b[5;30;43m")
                print(f"------{aspect}--------")
                stats.sort_stats(aspect).print_stats(8)
                stats.sort_stats(aspect).print_callers(8)
                print("\x1b[0m")
            print("\x1b[0m")
            self.prof.enable()
