"""Gestion des taches (VOD, live) : file d'attente, progression, journal."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .utils import log


@dataclass
class Job:
    kind: str
    url: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "en attente"      # en attente | en cours | termine | erreur | arrete
    progress: float = 0.0
    step: str = ""
    logs: list[str] = field(default_factory=list)
    clips: list[str] = field(default_factory=list)
    streamer: str = ""
    created: float = field(default_factory=time.time)
    stop_event: threading.Event = field(default_factory=threading.Event)
    options: dict = field(default_factory=dict)

    def say(self, msg: str, progress: float | None = None) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.logs.append(f"{stamp}  {msg}")
        self.logs = self.logs[-300:]
        self.step = msg
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        log.info("[%s] %s", self.id, msg)

    @property
    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "url": self.url, "status": self.status,
                "progress": round(self.progress, 3), "step": self.step, "logs": self.logs[-60:],
                "clips": self.clips, "streamer": self.streamer, "created": self.created}


class JobManager:
    """Les VOD passent une par une (le GPU est partage) ; les lives tournent en parallele
    mais leur montage passe aussi par le verrou GPU."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.gpu_lock = threading.Lock()
        self._vod_queue: list[tuple[Job, Callable]] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._vod_worker, daemon=True).start()

    def submit(self, job: Job, fn: Callable[[Job], None], queued: bool = True) -> Job:
        self.jobs[job.id] = job
        if queued:
            with self._cv:
                self._vod_queue.append((job, fn))
                self._cv.notify()
        else:
            threading.Thread(target=self._run, args=(job, fn), daemon=True).start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], None]) -> None:
        job.status = "en cours"
        try:
            fn(job)
            if job.status == "en cours":
                job.status = "arrete" if job.stopped else "termine"
                if not job.stopped:
                    job.progress = 1.0
        except Exception as e:
            job.status = "erreur"
            job.say(f"ERREUR : {e}")
            log.error(traceback.format_exc())

    def _vod_worker(self) -> None:
        while True:
            with self._cv:
                while not self._vod_queue:
                    self._cv.wait()
                job, fn = self._vod_queue.pop(0)
            if job.stopped:
                job.status = "arrete"
                continue
            self._run(job, fn)

    def stop(self, job_id: str) -> None:
        j = self.jobs.get(job_id)
        if j:
            j.stop_event.set()
            if j.status == "en attente":
                j.status = "arrete"


manager = JobManager()
