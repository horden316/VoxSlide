const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const loopbackHosts = new Set(["localhost", "127.0.0.1", "::1"]);

function resolveApiBaseUrl() {
  if (typeof window === "undefined") {
    return configuredApiBaseUrl.replace(/\/$/, "");
  }

  try {
    const apiUrl = new URL(configuredApiBaseUrl);
    if (loopbackHosts.has(apiUrl.hostname) && !loopbackHosts.has(window.location.hostname)) {
      apiUrl.hostname = window.location.hostname;
    }
    return apiUrl.toString().replace(/\/$/, "");
  } catch {
    return configuredApiBaseUrl.replace(/\/$/, "");
  }
}

export const API_BASE_URL = resolveApiBaseUrl();

export type Project = {
  id: number;
  title: string;
  original_pdf_path: string | null;
  output_video_path: string | null;
  status: string;
  created_at: string;
  updated_at: string;
};

export type SlidePage = {
  id: number;
  project_id: number;
  page_number: number;
  image_url: string;
  transcript: string;
  audio_url: string | null;
  audio_duration: number | null;
  created_at: string;
  updated_at: string;
};

export type Job = {
  id: number;
  project_id: number;
  type: string;
  status: string;
  progress: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type TtsConfig = {
  provider: string;
  model: string;
  default_voice: string;
  voices: {id: string; label: string}[];
};

export type TtsOptions = {
  voice?: string;
  language?: string;
  instruct?: string;
};

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async createProject(title: string) {
    return parseResponse<Project>(
      await fetch(`${API_BASE_URL}/api/projects`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({title}),
      }),
    );
  },
  async listProjects() {
    return parseResponse<Project[]>(await fetch(`${API_BASE_URL}/api/projects`));
  },
  async getPages(projectId: number) {
    return parseResponse<SlidePage[]>(await fetch(`${API_BASE_URL}/api/projects/${projectId}/pages`));
  },
  async getTtsConfig() {
    return parseResponse<TtsConfig>(await fetch(`${API_BASE_URL}/api/tts/config`));
  },
  async uploadPdf(projectId: number, file: File) {
    const form = new FormData();
    form.append("file", file);
    return parseResponse(await fetch(`${API_BASE_URL}/api/projects/${projectId}/upload-pdf`, {method: "POST", body: form}));
  },
  async saveTranscript(pageId: number, transcript: string) {
    return parseResponse<SlidePage>(
      await fetch(`${API_BASE_URL}/api/pages/${pageId}`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({transcript}),
      }),
    );
  },
  async generateAudio(pageId: number, options?: TtsOptions) {
    return parseResponse<SlidePage>(
      await fetch(`${API_BASE_URL}/api/pages/${pageId}/generate-audio`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(options || {}),
      }),
    );
  },
  async generateAudioJob(pageId: number, options?: TtsOptions) {
    return parseResponse<{job_id: number}>(
      await fetch(`${API_BASE_URL}/api/pages/${pageId}/generate-audio-job`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(options || {}),
      }),
    );
  },
  async renderVideo(projectId: number, options?: TtsOptions, forceRegenerate = false) {
    return parseResponse<{job_id: number}>(
      await fetch(`${API_BASE_URL}/api/projects/${projectId}/render-video`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...(options || {}), force_regenerate: forceRegenerate}),
      }),
    );
  },
  async getJob(jobId: number) {
    return parseResponse<Job>(await fetch(`${API_BASE_URL}/api/jobs/${jobId}`));
  },
};
