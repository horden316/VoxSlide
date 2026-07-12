"use client";

import {useEffect, useMemo, useRef, useState} from "react";
import {ChevronDown, Dices, Download, FileText, FileUp, Play, Plus, RefreshCw, RotateCcw, Save, SlidersHorizontal, Upload} from "lucide-react";
import {API_BASE_URL, api} from "@/lib/api";
import type {Job, Project, SlidePage, TtsConfig, TtsOptions, TtsParamValue} from "@/lib/api";

type AudioJobState = {
  id: number;
  pageId: number;
  pageNumber: number;
  status: string;
  progress: number;
  seed?: number;
};

const tonePresets = [
  {
    id: "default",
    label: "Speaker default (most stable)",
    instruct: "",
  },
  {
    id: "professional",
    label: "Professional presentation",
    instruct: "Speak in a clear, calm, professional presentation tone, consistent from start to finish.",
  },
  {
    id: "teaching",
    label: "Teaching explanation",
    instruct:
      "Speak in a patient, clear, and approachable teaching style. Emphasize important concepts gently and make the explanation easy to understand.",
  },
  {
    id: "warm",
    label: "Warm and friendly",
    instruct:
      "Speak in a warm, friendly, and natural style. Keep the voice soft but clear, as if explaining the content to colleagues.",
  },
  {
    id: "formal",
    label: "Formal and steady",
    instruct: "Speak in a calm, steady, formal tone, consistent throughout.",
  },
  {
    id: "energetic",
    label: "Energetic and natural",
    instruct:
      "Speak in an energetic, natural, and engaging style. Keep the pronunciation clear and avoid an exaggerated advertising tone.",
  },
];

const speedPresets = [
  {
    id: "slow",
    label: "Slower",
    instruct: "Use a slightly slower pace with natural pauses between sentences.",
  },
  {
    id: "medium",
    label: "Medium (no prompt)",
    instruct: "",
  },
  {
    id: "fast",
    label: "Slightly faster",
    instruct: "Use a slightly faster pace while keeping every sentence clear and easy to understand.",
  },
];

type TtsParamField = {
  key: string;
  label: string;
  step: number;
};

const ttsParamGroups: {title: string; fields: TtsParamField[]}[] = [
  {
    title: "Sampling (semantic layer)",
    fields: [
      {key: "temperature", label: "Temperature", step: 0.05},
      {key: "top_p", label: "Top-p", step: 0.05},
      {key: "top_k", label: "Top-k", step: 1},
      {key: "repetition_penalty", label: "Repetition penalty", step: 0.01},
      {key: "seed", label: "Seed", step: 1},
      {key: "max_new_tokens", label: "Max new tokens", step: 32},
    ],
  },
  {
    title: "Subtalker sampling (acoustic layer)",
    fields: [
      {key: "subtalker_temperature", label: "Temperature", step: 0.05},
      {key: "subtalker_top_p", label: "Top-p", step: 0.05},
      {key: "subtalker_top_k", label: "Top-k", step: 1},
    ],
  },
  {
    title: "Chunking",
    fields: [
      {key: "max_chars_per_chunk", label: "Max chars / chunk", step: 10},
      {key: "min_chunk_chars", label: "Min chunk chars", step: 1},
    ],
  },
  {
    title: "Pauses (ms)",
    fields: [
      {key: "sentence_gap_ms", label: "Sentence gap", step: 50},
      {key: "semicolon_gap_ms", label: "Semicolon gap", step: 50},
      {key: "paragraph_gap_ms", label: "Paragraph gap", step: 50},
      {key: "wrap_gap_ms", label: "Wrap gap", step: 50},
      {key: "pause_default_ms", label: "[pause] default", step: 100},
    ],
  },
  {
    title: "Audio trim",
    fields: [
      {key: "trim_threshold_db", label: "Trim threshold (dB)", step: 1},
      {key: "trim_pad_ms", label: "Trim pad (ms)", step: 5},
      {key: "edge_fade_ms", label: "Edge fade (ms)", step: 5},
    ],
  },
];

const booleanParamFields = [
  {key: "do_sample", label: "do_sample"},
  {key: "subtalker_dosample", label: "subtalker_dosample"},
];

// Mirrors the docker-compose defaults; only used when /api/tts/config cannot reach the TTS service.
const fallbackTtsDefaults: Record<string, TtsParamValue> = {
  seed: 316,
  do_sample: true,
  top_k: 10,
  top_p: 0.8,
  temperature: 0.6,
  repetition_penalty: 1.05,
  subtalker_dosample: true,
  subtalker_top_k: 10,
  subtalker_top_p: 0.8,
  subtalker_temperature: 0.6,
  max_new_tokens: 1024,
  max_chars_per_chunk: 200,
  min_chunk_chars: 80,
  sentence_gap_ms: 700,
  semicolon_gap_ms: 350,
  paragraph_gap_ms: 1000,
  wrap_gap_ms: 150,
  pause_default_ms: 1000,
  trim_threshold_db: -42,
  trim_pad_ms: 15,
  edge_fade_ms: 10,
};

const languageOptions = [
  {id: "", label: "Speaker default"},
  {id: "Chinese", label: "Chinese"},
  {id: "English", label: "English"},
  {id: "Japanese", label: "Japanese"},
  {id: "Korean", label: "Korean"},
];

function buildInstruct(toneId: string, speedId: string) {
  const tone = tonePresets.find((preset) => preset.id === toneId) || tonePresets[0];
  const speed = speedPresets.find((preset) => preset.id === speedId) || speedPresets[1];
  return [tone.instruct, speed.instruct].filter(Boolean).join(" ");
}

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
  const [selectedLanguage, setSelectedLanguage] = useState("");
  const [selectedTonePreset, setSelectedTonePreset] = useState(tonePresets[0].id);
  const [selectedSpeedPreset, setSelectedSpeedPreset] = useState(speedPresets[1].id);
  const [voiceInstruct, setVoiceInstruct] = useState(() => buildInstruct(tonePresets[0].id, speedPresets[1].id));
  const [audioJob, setAudioJob] = useState<AudioJobState | null>(null);
  const [forceRegenerate, setForceRegenerate] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [paramOverrides, setParamOverrides] = useState<Record<string, string>>({});
  const transcriptRefs = useRef<Record<number, HTMLTextAreaElement | null>>({});

  const downloadUrl = useMemo(
    () => (selectedProject ? `${API_BASE_URL}/api/projects/${selectedProject.id}/download` : "#"),
    [selectedProject],
  );
  const subtitleDownloadUrl = useMemo(
    () => (selectedProject ? `${API_BASE_URL}/api/projects/${selectedProject.id}/download-srt` : "#"),
    [selectedProject],
  );
  const hasCompletedVideo = selectedProject?.status === "completed" || jobStatus === "completed";

  function insertPauseMarker(page: SlidePage) {
    const marker = "[pause]";
    const textarea = transcriptRefs.current[page.id];
    const start = textarea?.selectionStart ?? page.transcript.length;
    const end = textarea?.selectionEnd ?? start;
    const transcript = `${page.transcript.slice(0, start)}${marker}${page.transcript.slice(end)}`;
    setPages((current) => current.map((item) => (item.id === page.id ? {...item, transcript} : item)));
    requestAnimationFrame(() => {
      if (!textarea) return;
      textarea.focus();
      const cursor = start + marker.length;
      textarea.setSelectionRange(cursor, cursor);
    });
  }

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

  const ttsDefaults = ttsConfig?.params ?? fallbackTtsDefaults;
  const overrideCount = Object.values(paramOverrides).filter((value) => value.trim() !== "").length;

  function setParamOverride(key: string, value: string) {
    setParamOverrides((current) => {
      const next = {...current};
      if (value === "") {
        delete next[key];
      } else {
        next[key] = value;
      }
      return next;
    });
  }

  function collectTtsParams(): Record<string, TtsParamValue> | undefined {
    const params: Record<string, TtsParamValue> = {};
    for (const [key, raw] of Object.entries(paramOverrides)) {
      const text = raw.trim();
      if (!text) continue;
      if (typeof ttsDefaults[key] === "boolean") {
        params[key] = text === "true";
      } else {
        const value = Number(text);
        if (Number.isFinite(value)) params[key] = value;
      }
    }
    return Object.keys(params).length > 0 ? params : undefined;
  }

  function ttsOptions(): TtsOptions {
    return {
      voice: selectedVoice || undefined,
      language: selectedLanguage || undefined,
      instruct: voiceInstruct.trim() || undefined,
      tts_params: collectTtsParams(),
    };
  }

  function updateTonePreset(presetId: string) {
    setSelectedTonePreset(presetId);
    setVoiceInstruct(buildInstruct(presetId, selectedSpeedPreset));
  }

  function updateSpeedPreset(presetId: string) {
    setSelectedSpeedPreset(presetId);
    setVoiceInstruct(buildInstruct(selectedTonePreset, presetId));
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
          const seedNote = audioJob.seed !== undefined ? ` (seed ${audioJob.seed})` : "";
          setMessage(job.status === "completed" ? `Page ${audioJob.pageNumber} audio generated${seedNote}.` : job.error_message || "Audio generation failed.");
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

  async function generateAudio(page: SlidePage, rerollSeed?: number) {
    setMessage("");
    if (!page.transcript.trim()) {
      setMessage(`Page ${page.page_number} needs a transcript before audio generation.`);
      return;
    }
    try {
      await api.saveTranscript(page.id, page.transcript);
      const options = ttsOptions();
      if (rerollSeed !== undefined) {
        options.tts_params = {...options.tts_params, seed: rerollSeed};
      }
      const job = await api.generateAudioJob(page.id, options);
      setAudioJob({id: job.job_id, pageId: page.id, pageNumber: page.page_number, status: "queued", progress: 0, seed: rerollSeed});
      setMessage(
        rerollSeed !== undefined
          ? `Page ${page.page_number} rerolling with seed ${rerollSeed}.`
          : `Page ${page.page_number} audio generation started.`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Audio generation failed.");
    }
  }

  function rerollAudio(page: SlidePage) {
    return generateAudio(page, Math.floor(Math.random() * 1_000_000));
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
      const job = await api.renderVideo(selectedProject.id, ttsOptions(), forceRegenerate);
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
              <button className="text-slate-500" onClick={() => refreshProjects()} title="Refresh projects">
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
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium">
                  <Upload size={16} />
                  <span>Upload PDF</span>
                  <input className="hidden" type="file" accept="application/pdf" onChange={(event) => uploadPdf(event.target.files?.[0] || null)} />
                </label>
                <label
                  className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium"
                  title="Ignore existing audio and re-synthesize every page with the current voice settings."
                >
                  <input
                    type="checkbox"
                    className="accent-slate-900"
                    checked={forceRegenerate}
                    onChange={(event) => setForceRegenerate(event.target.checked)}
                  />
                  <span>Regenerate all audio</span>
                </label>
                <button className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white" onClick={renderVideo}>
                  <FileUp size={16} />
                  <span>Render video</span>
                </button>
                {hasCompletedVideo && (
                  <>
                    <a className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium" href={downloadUrl}>
                      <Download size={16} />
                      <span>Video</span>
                    </a>
                    <a className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium" href={subtitleDownloadUrl}>
                      <FileText size={16} />
                      <span>SRT</span>
                    </a>
                  </>
                )}
              </div>
            )}
          </div>

          {selectedProject && ttsConfig && (
            <div className="mb-4 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-slate-900">Voice style</div>
                  <div className="text-xs text-slate-500">
                    {ttsConfig.provider} / {ttsConfig.model}
                  </div>
                </div>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                  <label className="text-xs font-medium text-slate-600">
                    Voice
                    <select
                      className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 outline-none focus:border-slate-700"
                      value={selectedVoice}
                      onChange={(event) => setSelectedVoice(event.target.value)}
                    >
                      {ttsConfig.voices.map((voice) => (
                        <option key={voice.id} value={voice.id}>
                          {voice.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-medium text-slate-600">
                    Language
                    <select
                      className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 outline-none focus:border-slate-700"
                      value={selectedLanguage}
                      onChange={(event) => setSelectedLanguage(event.target.value)}
                    >
                      {languageOptions.map((language) => (
                        <option key={language.id} value={language.id}>
                          {language.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-medium text-slate-600">
                    Tone preset
                    <select
                      className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 outline-none focus:border-slate-700"
                      value={selectedTonePreset}
                      onChange={(event) => updateTonePreset(event.target.value)}
                    >
                      {tonePresets.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-medium text-slate-600">
                    Speed prompt
                    <select
                      className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 outline-none focus:border-slate-700"
                      value={selectedSpeedPreset}
                      onChange={(event) => updateSpeedPreset(event.target.value)}
                    >
                      {speedPresets.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
              <label className="block text-xs font-medium text-slate-600">
                Instruct prompt
                <textarea
                  className="mt-1 min-h-24 w-full resize-y rounded-md border border-slate-300 p-3 text-sm text-slate-900 outline-none focus:border-slate-700"
                  value={voiceInstruct}
                  onChange={(event) => setVoiceInstruct(event.target.value)}
                  placeholder={
                    ttsConfig?.speaker_instructs?.[selectedVoice]
                      ? `Empty = speaker default: ${ttsConfig.speaker_instructs[selectedVoice]} (most stable). Keep custom instructions short and positive.`
                      : "Empty = speaker's built-in default (most stable). Keep custom instructions short and positive."
                  }
                />
              </label>

              <div className="mt-3 border-t border-slate-200 pt-3">
                <button
                  className="inline-flex items-center gap-2 text-xs font-medium text-slate-600 hover:text-slate-900"
                  onClick={() => setAdvancedOpen((open) => !open)}
                >
                  <SlidersHorizontal size={14} />
                  <span>Advanced TTS parameters</span>
                  {overrideCount > 0 && (
                    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-semibold text-amber-800">{overrideCount} modified</span>
                  )}
                  <ChevronDown size={14} className={`transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
                </button>
                {advancedOpen && (
                  <div className="mt-3">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-xs text-slate-500">
                        Empty fields use the server defaults shown as placeholders. Overrides apply to page audio and video rendering.
                      </p>
                      <button
                        className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:border-slate-500 hover:text-slate-900 disabled:opacity-50"
                        onClick={() => setParamOverrides({})}
                        disabled={overrideCount === 0}
                      >
                        <RotateCcw size={12} />
                        <span>Reset all</span>
                      </button>
                    </div>
                    <div className="mb-3 flex flex-wrap gap-4">
                      {booleanParamFields.map((field) => (
                        <label key={field.key} className="text-xs font-medium text-slate-600">
                          {field.label}
                          <select
                            className="mt-1 block rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 outline-none focus:border-slate-700"
                            value={paramOverrides[field.key] ?? ""}
                            onChange={(event) => setParamOverride(field.key, event.target.value)}
                          >
                            <option value="">Default ({String(ttsDefaults[field.key])})</option>
                            <option value="true">true</option>
                            <option value="false">false</option>
                          </select>
                        </label>
                      ))}
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                      {ttsParamGroups.map((group) => (
                        <fieldset key={group.title} className="rounded-md border border-slate-200 p-3">
                          <legend className="px-1 text-xs font-semibold text-slate-700">{group.title}</legend>
                          <div className="grid gap-2">
                            {group.fields.map((field) => (
                              <label key={field.key} className="flex items-center justify-between gap-2 text-xs text-slate-600">
                                <span>{field.label}</span>
                                <input
                                  type="number"
                                  step={field.step}
                                  className={`w-24 rounded-md border px-2 py-1 text-right text-sm text-slate-900 outline-none focus:border-slate-700 ${
                                    paramOverrides[field.key] ? "border-amber-400 bg-amber-50" : "border-slate-300"
                                  }`}
                                  placeholder={String(ttsDefaults[field.key] ?? "")}
                                  value={paramOverrides[field.key] ?? ""}
                                  onChange={(event) => setParamOverride(field.key, event.target.value)}
                                />
                              </label>
                            ))}
                          </div>
                        </fieldset>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

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
                <div className="flex flex-wrap gap-2">
                  <a
                    className="inline-flex items-center gap-2 rounded-md bg-emerald-700 px-3 py-2 text-sm font-medium text-white"
                    href={downloadUrl}
                  >
                    <Download size={16} />
                    <span>Download video</span>
                  </a>
                  <a
                    className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium"
                    href={subtitleDownloadUrl}
                  >
                    <FileText size={16} />
                    <span>Download SRT</span>
                  </a>
                </div>
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
                        <button
                          className="inline-flex items-center gap-2 rounded-md border border-emerald-700 px-3 py-2 text-sm text-emerald-800 disabled:opacity-60"
                          onClick={() => rerollAudio(page)}
                          disabled={Boolean(audioJob) || !page.audio_url}
                          title="Regenerate this page with a random seed to get a different take."
                        >
                          <Dices size={16} />
                          <span>Reroll</span>
                        </button>
                      </div>
                    </div>
                    <div className="mb-1 flex justify-end">
                      <button
                        className="inline-flex items-center gap-1 rounded border border-dashed border-slate-300 px-2 py-1 font-mono text-xs text-slate-500 hover:border-slate-500 hover:text-slate-900"
                        onClick={() => insertPauseMarker(page)}
                        title="Insert a pause marker at the cursor. Edit it to [pause:1500] for a custom length in milliseconds."
                      >
                        <Plus size={12} />
                        <span>[pause]</span>
                      </button>
                    </div>
                    <textarea
                      ref={(element) => {
                        transcriptRefs.current[page.id] = element;
                      }}
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
