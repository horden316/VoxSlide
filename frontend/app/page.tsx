"use client";

import {useEffect, useMemo, useState} from "react";
import {Download, FileUp, Play, Plus, RefreshCw, Save, Upload} from "lucide-react";
import {API_BASE_URL, api} from "@/lib/api";
import type {Job, Project, SlidePage, TtsConfig} from "@/lib/api";

type AudioJobState = {
  id: number;
  pageId: number;
  pageNumber: number;
  status: string;
  progress: number;
};

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [pages, setPages] = useState<SlidePage[]>([]);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [jobId, setJobId] = useState<number | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobStatus, setJobStatus] = useState("");
  const [ttsConfig, setTtsConfig] = useState<TtsConfig | null>(null);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [audioJob, setAudioJob] = useState<AudioJobState | null>(null);

  const downloadUrl = useMemo(
    () => (selectedProject ? `${API_BASE_URL}/api/projects/${selectedProject.id}/download` : "#"),
    [selectedProject],
  );
  const hasCompletedVideo = selectedProject?.status === "completed" || jobStatus === "completed";

  function audioSource(page: SlidePage) {
    if (!page.audio_url) return "";
    const version = encodeURIComponent(`${page.updated_at}-${page.audio_duration ?? ""}`);
    return `${API_BASE_URL}${page.audio_url}?v=${version}`;
  }

  async function refreshProjects(preferredProjectId = selectedProject?.id) {
    const loaded = await api.listProjects();
    setProjects(loaded);
    if (preferredProjectId) {
      setSelectedProject(loaded.find((project) => project.id === preferredProjectId) || selectedProject);
    } else if (loaded.length > 0) {
      setSelectedProject(loaded[0]);
    }
  }

  async function refreshPages(projectId = selectedProject?.id) {
    if (!projectId) return;
    setPages(await api.getPages(projectId));
  }

  useEffect(() => {
    refreshProjects().catch((error) => setMessage(error.message));
    api
      .getTtsConfig()
      .then((config) => {
        setTtsConfig(config);
        setSelectedVoice(config.default_voice);
      })
      .catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    if (selectedProject) {
      refreshPages(selectedProject.id).catch((error) => setMessage(error.message));
    }
  }, [selectedProject?.id]);

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await api.getJob(jobId);
        setJobProgress(job.progress);
        setJobStatus(job.status);
        if (job.status === "completed" || job.status === "failed") {
          window.clearInterval(timer);
          setMessage(job.status === "completed" ? "" : job.error_message || "Render failed.");
          await refreshProjects();
        }
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not load job.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId]);

  useEffect(() => {
    if (!audioJob) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await api.getJob(audioJob.id);
        setAudioJob((current) => (current && current.id === job.id ? jobToAudioState(job, current) : current));
        if (job.status === "completed" || job.status === "failed") {
          window.clearInterval(timer);
          setMessage(job.status === "completed" ? `Page ${audioJob.pageNumber} audio generated.` : job.error_message || "Audio generation failed.");
          if (job.status === "completed") {
            await refreshPages();
          }
          setAudioJob(null);
        }
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not load audio job.");
        window.clearInterval(timer);
        setAudioJob(null);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [audioJob?.id]);

  function jobToAudioState(job: Job, current: AudioJobState): AudioJobState {
    return {
      ...current,
      status: job.status,
      progress: job.progress,
    };
  }

  async function createProject() {
    if (!title.trim()) return;
    setBusy(true);
    setMessage("");
    try {
      const project = await api.createProject(title);
      setTitle("");
      setSelectedProject(project);
      await refreshProjects(project.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Project creation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function uploadPdf(file: File | null) {
    if (!selectedProject || !file) return;
    setBusy(true);
    setMessage("");
    try {
      await api.uploadPdf(selectedProject.id, file);
      await refreshPages(selectedProject.id);
      await refreshProjects();
      setMessage("PDF uploaded and converted.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "PDF upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function saveTranscript(page: SlidePage) {
    setMessage("");
    try {
      const updated = await api.saveTranscript(page.id, page.transcript);
      setPages((current) => current.map((item) => (item.id === page.id ? updated : item)));
      setMessage(`Page ${page.page_number} saved.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Save failed.");
    }
  }

  async function generateAudio(page: SlidePage) {
    setMessage("");
    if (!page.transcript.trim()) {
      setMessage(`Page ${page.page_number} needs a transcript before audio generation.`);
      return;
    }
    try {
      await api.saveTranscript(page.id, page.transcript);
      const job = await api.generateAudioJob(page.id, selectedVoice);
      setAudioJob({id: job.job_id, pageId: page.id, pageNumber: page.page_number, status: "queued", progress: 0});
      setMessage(`Page ${page.page_number} audio generation started.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Audio generation failed.");
    }
  }

  async function renderVideo() {
    if (!selectedProject) return;
    setMessage("");
    const missingPages = pages.filter((page) => !page.transcript.trim()).map((page) => page.page_number);
    if (missingPages.length > 0) {
      setMessage(`Transcript is required for pages: ${missingPages.join(", ")}`);
      return;
    }
    setJobProgress(0);
    setJobStatus("queued");
    try {
      const savedPages = await Promise.all(pages.map((page) => api.saveTranscript(page.id, page.transcript)));
      setPages(savedPages);
      const job = await api.renderVideo(selectedProject.id, selectedVoice);
      setJobId(job.job_id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Render failed to start.");
    }
  }

  return (
    <main className="min-h-screen">
      <div className="mx-auto flex max-w-7xl gap-6 px-5 py-6">
        <aside className="w-80 shrink-0">
          <div className="mb-5">
            <h1 className="text-3xl font-semibold tracking-tight">VoxSlide</h1>
            <p className="mt-1 text-sm text-slate-600">PDF slides to narrated MP4.</p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <label className="text-sm font-medium text-slate-700">New project</label>
            <div className="mt-2 flex gap-2">
              <input
                className="min-w-0 flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-700"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Deck title"
              />
              <button
                className="grid h-10 w-10 place-items-center rounded-md bg-slate-900 text-white disabled:opacity-50"
                onClick={createProject}
                disabled={busy}
                title="Create project"
              >
                <Plus size={18} />
              </button>
            </div>
          </div>

          <div className="mt-4 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              <span className="text-sm font-medium">Projects</span>
              <button className="text-slate-500" onClick={refreshProjects} title="Refresh projects">
                <RefreshCw size={16} />
              </button>
            </div>
            {projects.map((project) => (
              <button
                key={project.id}
                className={`block w-full border-b border-slate-100 px-4 py-3 text-left text-sm ${
                  selectedProject?.id === project.id ? "bg-slate-100" : "bg-white"
                }`}
                onClick={() => setSelectedProject(project)}
              >
                <div className="font-medium">{project.title}</div>
                <div className="mt-1 text-xs text-slate-500">{project.status}</div>
              </button>
            ))}
          </div>
        </aside>

        <section className="min-w-0 flex-1">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
            <div>
              <div className="text-sm text-slate-500">Current project</div>
              <div className="text-xl font-semibold">{selectedProject?.title || "Create a project to begin"}</div>
            </div>
            {selectedProject && (
              <div className="flex items-center gap-2">
                {ttsConfig && (
                  <label className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium">
                    <span className="text-slate-500">Voice</span>
                    <select
                      className="bg-transparent text-sm outline-none"
                      value={selectedVoice}
                      onChange={(event) => setSelectedVoice(event.target.value)}
                      title={`TTS provider: ${ttsConfig.provider}, model: ${ttsConfig.model}`}
                    >
                      {ttsConfig.voices.map((voice) => (
                        <option key={voice.id} value={voice.id}>
                          {voice.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium">
                  <Upload size={16} />
                  <span>Upload PDF</span>
                  <input className="hidden" type="file" accept="application/pdf" onChange={(event) => uploadPdf(event.target.files?.[0] || null)} />
                </label>
                <button className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white" onClick={renderVideo}>
                  <FileUp size={16} />
                  <span>Render video</span>
                </button>
                {hasCompletedVideo && (
                  <a className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium" href={downloadUrl}>
                    <Download size={16} />
                    <span>Download</span>
                  </a>
                )}
              </div>
            )}
          </div>

          {(message || jobStatus) && (
            <div className="mb-4 rounded-lg border border-slate-200 bg-white p-4 text-sm shadow-sm">
              {message && <div>{message}</div>}
              {jobStatus && jobStatus !== "completed" && (
                <div className="mt-2">
                  <div className="mb-1 flex justify-between text-xs text-slate-500">
                    <span>{jobStatus}</span>
                    <span>{jobProgress}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-200">
                    <div className="h-2 rounded-full bg-emerald-600" style={{width: `${jobProgress}%`}} />
                  </div>
                </div>
              )}
              {jobStatus === "completed" && selectedProject && (
                <a
                  className="inline-flex items-center gap-2 rounded-md bg-emerald-700 px-3 py-2 text-sm font-medium text-white"
                  href={downloadUrl}
                >
                  <Download size={16} />
                  <span>Download video</span>
                </a>
              )}
            </div>
          )}

          <div className="grid gap-4">
            {pages.map((page) => (
              <article key={page.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
                  <img
                    src={`${API_BASE_URL}${page.image_url}`}
                    alt={`Page ${page.page_number}`}
                    className="aspect-video w-full rounded-md border border-slate-200 object-contain"
                  />
                  <div className="min-w-0">
                    <div className="mb-2 flex items-center justify-between">
                      <h2 className="font-semibold">Page {page.page_number}</h2>
                      <div className="flex gap-2">
                        <button className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm" onClick={() => saveTranscript(page)}>
                          <Save size={16} />
                          <span>Save</span>
                        </button>
                        <button
                          className="inline-flex items-center gap-2 rounded-md bg-emerald-700 px-3 py-2 text-sm text-white disabled:opacity-60"
                          onClick={() => generateAudio(page)}
                          disabled={Boolean(audioJob)}
                        >
                          <Play size={16} />
                          <span>{audioJob?.pageId === page.id ? "Generating" : "Audio"}</span>
                        </button>
                      </div>
                    </div>
                    <textarea
                      className="min-h-32 w-full resize-y rounded-md border border-slate-300 p-3 text-sm outline-none focus:border-slate-700"
                      value={page.transcript}
                      onChange={(event) =>
                        setPages((current) => current.map((item) => (item.id === page.id ? {...item, transcript: event.target.value} : item)))
                      }
                      placeholder="Transcript for this slide"
                    />
                    {audioJob?.pageId === page.id && (
                      <div className="mt-3">
                        <div className="mb-1 flex justify-between text-xs text-slate-500">
                          <span>{audioJob.status}</span>
                          <span>{audioJob.progress}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-200">
                          <div className="h-2 rounded-full bg-emerald-600" style={{width: `${audioJob.progress}%`}} />
                        </div>
                      </div>
                    )}
                    {page.audio_url && (
                      <audio key={audioSource(page)} className="mt-3 w-full" controls src={audioSource(page)}>
                        <track kind="captions" />
                      </audio>
                    )}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
