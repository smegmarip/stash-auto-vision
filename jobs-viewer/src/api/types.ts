// ============================================
// Job Types
// ============================================

export type JobStatus = "queued" | "processing" | "completed" | "failed";

export interface JobSummary {
  job_id: string;
  service: "vision" | "faces" | "scenes";
  status: JobStatus;
  progress: number;
  source?: string;
  source_id?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  result_summary?: Record<string, unknown>;
  results?: JobResults;
}

export interface ListJobsResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface JobFilters {
  status?: JobStatus;
  service?: "vision" | "faces" | "scenes";
  source_id?: string;
  source?: string;
  start_date?: string;
  end_date?: string;
  include_results?: boolean;
  limit?: number;
  offset?: number;
}

// ============================================
// Vision API Types
// ============================================

export interface ServiceJobInfo {
  service: string;
  job_id?: string;
  status: string;
  progress: number;
  message?: string;
  error?: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  processing_mode: string;
  stage?: string;
  message?: string;
  services: ServiceJobInfo[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result_summary?: Record<string, unknown>;
  error?: string;
}

export interface JobResults {
  job_id: string;
  source_id: string;
  status: string;
  scenes?: ScenesResult;
  faces?: FacesResult;
  semantics?: Record<string, unknown>;
  objects?: Record<string, unknown>;
  metadata: JobMetadata;
}

export interface JobMetadata {
  processing_time_seconds: number;
  processing_mode: string;
  services_used?: Record<string, boolean>;
}

// ============================================
// Faces Types
// ============================================

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface Landmarks {
  left_eye: [number, number];
  right_eye: [number, number];
  nose: [number, number];
  mouth_left: [number, number];
  mouth_right: [number, number];
}

export interface Demographics {
  age: number;
  gender: "M" | "F";
  emotion: string;
}

export interface QualityComponents {
  size: number;
  pose: number;
  occlusion: number;
  sharpness: number;
}

export interface Quality {
  composite: number;
  components: QualityComponents;
}

export interface Occlusion {
  occluded: boolean;
  probability: number;
}

export interface Detection {
  frame_index: number;
  timestamp: number;
  bbox: BoundingBox;
  confidence: number;
  quality: Quality;
  pose: string;
  landmarks: Landmarks;
  enhanced: boolean;
  occlusion: Occlusion;
}

export interface Face {
  face_id: string;
  embedding: number[];
  demographics?: Demographics;
  detections: Detection[];
  representative_detection: Detection;
}

export interface FacesMetadata {
  source: string;
  total_frames: number;
  frames_processed: number;
  unique_faces: number;
  total_detections: number;
  processing_time_seconds: number;
  method: string;
  model: string;
}

export interface FacesResult {
  job_id: string;
  source_id: string;
  status: string;
  faces: Face[];
  metadata: FacesMetadata;
}

// ============================================
// Scenes Types
// ============================================

export interface Scene {
  scene_number: number;
  // Backend uses start_timestamp/end_timestamp
  start_time?: number;
  end_time?: number;
  start_timestamp?: number;
  end_timestamp?: number;
  start_frame: number;
  end_frame: number;
  duration: number;
}

export interface ScenesMetadata {
  video_path: string;
  // Backend may use different field names
  total_duration_seconds?: number;
  video_duration_seconds?: number;
  total_scenes?: number;
  processing_time_seconds?: number;
  detector?: string;
  detection_method?: string;
  threshold?: number;
  total_frames?: number;
  video_fps?: number;
}

export interface ScenesResult {
  job_id: string;
  source_id: string;
  status: string;
  scenes: Scene[];
  metadata: ScenesMetadata;
}

// ============================================
// WebSocket Types
// ============================================

export interface WSMessage {
  type: "CONNECTED" | "JOB_STATUS_UPDATE" | "PONG";
  jobId?: string;
  data?: unknown;
}

export interface WSJobStatusUpdate {
  type: "JOB_STATUS_UPDATE";
  jobId: string;
  data: {
    status: JobStatus;
    progress: number;
    message?: string;
    error?: string;
  };
}
