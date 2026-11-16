"""Nonblocking, process-wide project inference exclusion; no service changes."""
from pathlib import Path


class ProjectInferenceLock:
    def __enter__(self):
        path=Path(__file__).resolve().parents[1]/'runs'/'.inference.lock'
        path.parent.mkdir(exist_ok=True)
        self.file=path.open('a+b')
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            import os
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError('Another project inference request holds the sequential lock') from exc
        return self

    def __exit__(self,*args):
        import os
        try:
            self.file.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        finally:self.file.close()
