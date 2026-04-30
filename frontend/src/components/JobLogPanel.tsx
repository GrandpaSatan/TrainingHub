import { useEffect, useMemo, useState } from "react";
import { Activity, Square } from "lucide-react";
import { api, JobRecord } from "../api/client";

export type JobEvent = {
  id?: number;
  created_at?: number;
  event_type: string;
  level: string;
  message: string;
  data?: Record<string, unknown>;
};

export function JobLogPanel({ job, events: externalEvents }: { job?: JobRecord; events?: JobEvent[] }) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const visibleEvents = externalEvents || events;
  const terminal = useMemo(() => ["succeeded", "failed", "cancelled"].includes(job?.status || ""), [job]);

  useEffect(() => {
    if (externalEvents) {
      return;
    }
    setEvents([]);
    if (!job) {
      return;
    }
    const source = new EventSource(`/api/jobs/${job.job_id}/events`);
    source.onmessage = (event) => {
      setEvents((current) => [...current, JSON.parse(event.data)]);
    };
    [
      "queued",
      "started",
      "inference_shutdown",
      "worker_start",
      "training_start",
      "training_complete",
      "metric",
      "artifact",
      "succeeded",
      "failed",
      "cancelled",
      "worker_error",
    ].forEach((name) => {
      source.addEventListener(name, (event) => {
        setEvents((current) => [...current, JSON.parse((event as MessageEvent).data)]);
      });
    });
    source.addEventListener("close", () => source.close());
    return () => source.close();
  }, [externalEvents, job?.job_id]);

  async function cancelJob() {
    if (!job) {
      return;
    }
    await api.post(`/api/jobs/${job.job_id}/cancel`, {});
  }

  if (!job) {
    return <div className="empty">Select or start a job to view live events.</div>;
  }

  return (
    <section className="panel">
      <div className="panelHeader">
        <div>
          <h2>{job.job_id}</h2>
          <p>{job.job_type} - {job.status}</p>
        </div>
        <button className="iconButton danger" onClick={cancelJob} disabled={terminal} title="Cancel job">
          <Square size={16} />
        </button>
      </div>
      <div className="logStream" aria-live="polite">
        {visibleEvents.length === 0 ? (
          <div className="emptyInline"><Activity size={16} /> Waiting for events</div>
        ) : (
          visibleEvents.map((event, index) => (
            <div key={`${event.event_type}-${index}`} className={`logLine ${event.level}`}>
              <span>{event.event_type}</span>
              <p>{event.message}</p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
