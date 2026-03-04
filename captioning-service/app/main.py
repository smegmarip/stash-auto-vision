"""
Captioning Service - Main Application
FastAPI server for video captioning using JoyCaption VLM
"""

import os
import uuid
import time
import asyncio
import gc
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from .models import (
    AnalyzeCaptionsRequest,
    AnalyzeCaptionsResponse,
    JobStatusResponse,
    CaptionResults,
    CaptionOutcome,
    FrameCaption,
    CaptionTag,
    SceneCaptionSummary,
    CaptionMetadata,
    HealthResponse,
    JobStatus,
    PromptType,
    FrameSelectionMethod,
    TaxonomyResponse,
    TagTaxonomyNode,
    SceneSummaryData,
    PersonDetail,
    CinematographyInfo,
    VisualStyle,
    EnvironmentInfo
)
from .cache_manager import CacheManager
from .joycaption import JoyCaptionProcessor
from .prompt_templates import (
    get_prompt_template,
    parse_booru_tags,
    detect_tag_category,
    estimate_vram_usage
)
from .tag_aligner import TagAligner
from .hierarchical_tagger import HierarchicalTagger, ScoredTag
from .stash_client import StashClient
from .frame_client import FrameServerClient
from .scenes_client import ScenesServerClient
from .resource_client import ResourceManagerClient

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_SERVER_URL = os.getenv("FRAME_SERVER_URL", "http://frame-server:5001")
SCENES_SERVER_URL = os.getenv("SCENES_SERVER_URL", "http://scenes-service:5002")
RESOURCE_MANAGER_URL = os.getenv("RESOURCE_MANAGER_URL", "http://resource-manager:5007")
STASH_URL = os.getenv("STASH_URL", "http://localhost:9999")
STASH_API_KEY = os.getenv("STASH_API_KEY", "")
CAPTION_DEVICE = os.getenv("CAPTION_DEVICE", "cuda")
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
STUB_MODE = os.getenv("CAPTIONING_STUB_MODE", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
cache_manager: Optional[CacheManager] = None
joycaption: Optional[JoyCaptionProcessor] = None
tag_aligner: Optional[TagAligner] = None
hierarchical_tagger: Optional[HierarchicalTagger] = None
stash_client: Optional[StashClient] = None
frame_client: Optional[FrameServerClient] = None
resource_client: Optional[ResourceManagerClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cache_manager, joycaption, tag_aligner, hierarchical_tagger
    global stash_client, frame_client, resource_client

    logger.info("Starting Captioning Service...")

    if STUB_MODE:
        logger.warning("Running in STUB MODE - model will not be loaded")
        yield
        return

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="captions", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize JoyCaption processor (don't load model yet)
    joycaption = JoyCaptionProcessor(
        device=CAPTION_DEVICE,
        use_quantization=USE_QUANTIZATION
    )
    logger.info("JoyCaption processor initialized (model not loaded)")

    # Initialize clients
    frame_client = FrameServerClient(FRAME_SERVER_URL)
    resource_client = ResourceManagerClient(
        RESOURCE_MANAGER_URL,
        service_name="captioning-service"
    )
    stash_client = StashClient(STASH_URL, STASH_API_KEY)
    logger.info("Service clients initialized")

    # Try to load taxonomy from Stash
    try:
        if await stash_client.health_check():
            taxonomy = await stash_client.get_all_tags()
            tag_aligner = TagAligner(taxonomy)
            hierarchical_tagger = HierarchicalTagger(taxonomy)
            logger.info(f"Tag aligner initialized with {len(taxonomy)} tags")
            logger.info(f"Hierarchical tagger stats: {hierarchical_tagger.get_hierarchy_stats()}")
        else:
            logger.warning("Stash not available, tag alignment disabled")
            tag_aligner = TagAligner([])
            hierarchical_tagger = HierarchicalTagger([])
    except Exception as e:
        logger.warning(f"Failed to load taxonomy from Stash: {e}")
        tag_aligner = TagAligner([])
        hierarchical_tagger = HierarchicalTagger([])

    yield

    # Cleanup
    logger.info("Shutting down Captioning Service...")

    if joycaption and joycaption.model_loaded:
        joycaption.unload_model()

    if cache_manager:
        await cache_manager.disconnect()

    if frame_client:
        await frame_client.close()

    if resource_client:
        if resource_client.has_gpu_lease:
            await resource_client.release_gpu()
        await resource_client.close()

    if stash_client:
        await stash_client.close()

    logger.info("Captioning Service stopped")


app = FastAPI(
    title="Captioning Service",
    description="Video captioning using JoyCaption VLM",
    version="1.0.0",
    lifespan=lifespan
)


async def process_captioning_job(job_id: str, request: AnalyzeCaptionsRequest):
    """Background task to process captioning analysis"""
    start_time = time.time()
    gpu_wait_start = None
    gpu_wait_time = 0.0

    try:
        logger.info(f"Starting captioning job {job_id}")

        # Initialize job metadata
        metadata = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "stage": "initializing",
            "message": "Job queued",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source_id": request.source_id,
            "source": request.source,
        }

        # Generate cache key
        cache_params = {
            "prompt_type": request.parameters.prompt_type.value,
            "frame_selection": request.parameters.frame_selection.value,
            "min_confidence": request.parameters.min_confidence,
            "max_tags": request.parameters.max_tags_per_frame,
            "align_taxonomy": request.parameters.align_to_taxonomy,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)
        metadata["cache_key"] = cache_key

        await cache_manager.cache_job_metadata(job_id, cache_key, metadata)

        # Step 1: Request GPU access from resource manager
        logger.info("Requesting GPU access...")
        await cache_manager.update_job_status(
            job_id,
            JobStatus.WAITING_FOR_GPU.value,
            progress=0.05,
            stage="waiting_for_gpu",
            message="Waiting for GPU access"
        )

        vram_needed = estimate_vram_usage(
            batch_size=request.parameters.batch_size,
            use_quantization=request.parameters.use_quantization
        ) * 1024  # Convert GB to MB

        gpu_wait_start = time.time()

        try:
            if await resource_client.health_check():
                gpu_request = await resource_client.request_gpu(
                    vram_mb=vram_needed,
                    priority=5
                )

                if not gpu_request.get("granted"):
                    # Wait for GPU
                    await resource_client.wait_for_gpu(
                        gpu_request["request_id"],
                        max_wait=300.0
                    )
            else:
                logger.warning("Resource manager not available, proceeding without GPU management")
        except Exception as e:
            logger.warning(f"GPU management failed: {e}, proceeding anyway")

        gpu_wait_time = time.time() - gpu_wait_start

        # Step 2: Load model if needed
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.1,
            stage="loading_model",
            message="Loading JoyCaption model"
        )

        if not joycaption.model_loaded:
            joycaption.load_model()

        # Step 3: Get scene boundaries if needed
        scene_boundaries = None
        scenes_client = ScenesServerClient(SCENES_SERVER_URL)

        # Determine if we need scene boundaries
        needs_scenes = (
            request.parameters.frame_selection == FrameSelectionMethod.SCENE_BASED or
            request.scenes_job_id
        )

        if needs_scenes:
            if request.scenes_job_id:
                # Use provided scenes job
                await cache_manager.update_job_status(
                    job_id,
                    JobStatus.PROCESSING.value,
                    progress=0.15,
                    stage="fetching_scenes",
                    message="Fetching scene boundaries"
                )

                try:
                    scene_boundaries = await scenes_client.get_scene_boundaries(request.scenes_job_id)
                    logger.info(f"Retrieved {len(scene_boundaries)} scene boundaries from job {request.scenes_job_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch scene boundaries from {request.scenes_job_id}: {e}")
                    # If scene_based but fetch failed, trigger detection
                    if request.parameters.frame_selection == FrameSelectionMethod.SCENE_BASED:
                        logger.info("Triggering automatic scene detection...")
                        request.scenes_job_id = None  # Clear invalid job_id

            # Auto-trigger scene detection if scene_based and no valid boundaries yet
            if request.parameters.frame_selection == FrameSelectionMethod.SCENE_BASED and not scene_boundaries:
                await cache_manager.update_job_status(
                    job_id,
                    JobStatus.PROCESSING.value,
                    progress=0.12,
                    stage="detecting_scenes",
                    message="Running automatic scene detection"
                )

                try:
                    scenes_job_id, scene_boundaries = await scenes_client.analyze_scenes(
                        video_path=request.source,
                        threshold=27.0,
                        timeout=300.0
                    )
                    logger.info(f"Auto scene detection completed: {len(scene_boundaries)} scenes (job: {scenes_job_id})")
                except Exception as e:
                    logger.warning(f"Auto scene detection failed: {e}, falling back to interval extraction")
                    scene_boundaries = None

        # Step 4: Extract frames
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.2,
            stage="extracting_frames",
            message="Extracting frames from video"
        )

        # Choose extraction method based on frame_selection parameter
        if (request.parameters.frame_selection == FrameSelectionMethod.SPRITE_SHEET and
            request.parameters.sprite_vtt_url and
            request.parameters.sprite_image_url):
            # Use sprite sheet extraction (ultra-fast, bypasses video decoding)
            logger.info("Using sprite sheet extraction")
            frame_result = await frame_client.extract_sprites(
                video_path=request.source,
                sprite_vtt_url=request.parameters.sprite_vtt_url,
                sprite_image_url=request.parameters.sprite_image_url,
                frames_per_scene=request.parameters.frames_per_scene,
                scene_boundaries=scene_boundaries,
                select_sharpest=request.parameters.select_sharpest
            )
        else:
            # Use video-based extraction
            frame_result = await frame_client.extract_frames(
                video_path=request.source,
                sampling_interval=request.parameters.sampling_interval,
                scene_boundaries=scene_boundaries,
                frames_per_scene=request.parameters.frames_per_scene,
                select_sharpest=request.parameters.select_sharpest,
                candidate_multiplier=request.parameters.sharpness_candidate_multiplier
            )

        if not frame_result or not frame_result.get("frames"):
            raise RuntimeError("No frames extracted from video")

        frame_metadata = frame_result["frames"]
        logger.info(f"Extracted {len(frame_metadata)} frames")

        # Step 5: Load frame images
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.3,
            stage="loading_frames",
            message=f"Loading {len(frame_metadata)} frames"
        )

        timestamps = [f["timestamp"] for f in frame_metadata]
        frames = await frame_client.get_frames_batch(request.source, timestamps)

        valid_frames = []
        valid_metadata = []
        for frame, meta in zip(frames, frame_metadata):
            if frame is not None:
                valid_frames.append(frame)
                valid_metadata.append(meta)

        if not valid_frames:
            raise RuntimeError("Failed to load any frames")

        logger.info(f"Loaded {len(valid_frames)}/{len(frame_metadata)} frames")

        # Step 6: Generate captions
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.4,
            stage="captioning",
            message=f"Generating captions for {len(valid_frames)} frames"
        )

        prompt = request.parameters.custom_prompt or get_prompt_template(
            request.parameters.prompt_type
        )

        def progress_callback(current, total):
            progress = 0.4 + (0.4 * current / total)
            asyncio.create_task(
                cache_manager.update_job_status(
                    job_id,
                    JobStatus.PROCESSING.value,
                    progress=progress,
                    stage="captioning",
                    message=f"Captioning frame {current}/{total}"
                )
            )

        captions = joycaption.caption_images_batch(
            images=valid_frames,
            prompt=prompt,
            batch_size=request.parameters.batch_size,
            progress_callback=progress_callback
        )

        # Step 7: Parse and align tags
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.85,
            stage="aligning_tags",
            message="Aligning tags to taxonomy"
        )

        frame_results = []
        is_scene_summary = request.parameters.prompt_type == PromptType.SCENE_SUMMARY

        for meta, caption in zip(valid_metadata, captions):
            # Parse summary data for SCENE_SUMMARY prompt type
            summary_data = None
            if is_scene_summary:
                summary_data = parse_scene_summary_json(caption)
                if summary_data:
                    logger.debug(f"Parsed SCENE_SUMMARY for frame at {meta['timestamp']:.2f}s")

            # Parse booru-style tags from caption (for non-JSON captions)
            # For SCENE_SUMMARY, extract tags from parsed data instead
            if is_scene_summary and summary_data:
                # Extract tags from structured summary
                raw_tags = []
                if summary_data.setting:
                    raw_tags.append(summary_data.setting.replace(" ", "_"))
                if summary_data.locale:
                    raw_tags.append(summary_data.locale.replace(" ", "_"))
                raw_tags.extend([obj.replace(" ", "_") for obj in summary_data.objects])
                raw_tags.extend([act.replace(" ", "_") for act in summary_data.activities])
                raw_tags.extend([item.replace(" ", "_") for item in summary_data.attire])
                if summary_data.mood:
                    raw_tags.append(summary_data.mood.replace(" ", "_"))
                if summary_data.genre:
                    raw_tags.append(summary_data.genre.replace(" ", "_"))
                if summary_data.cinematography and summary_data.cinematography.shot_type:
                    raw_tags.append(summary_data.cinematography.shot_type)
                if summary_data.environment and summary_data.environment.time_of_day:
                    raw_tags.append(summary_data.environment.time_of_day)
            else:
                raw_tags = parse_booru_tags(caption)

            # Align to taxonomy if enabled
            if request.parameters.align_to_taxonomy:
                if request.parameters.use_hierarchical_scoring and hierarchical_tagger:
                    # Use DFS hierarchical scoring
                    scored_tags = hierarchical_tagger.score_text(caption)
                    aligned_tags = hierarchical_tagger.convert_to_caption_tags(
                        scored_tags[:request.parameters.max_tags_per_frame],
                        source="hierarchical"
                    )
                elif tag_aligner:
                    # Use fuzzy string matching
                    aligned_tags = tag_aligner.align_tags(
                        raw_tags,
                        min_confidence=request.parameters.min_confidence
                    )
                else:
                    aligned_tags = []
            else:
                aligned_tags = [
                    CaptionTag(
                        tag=tag,
                        confidence=max(0.5, 1.0 - (i * 0.02)),
                        source="joycaption",
                        category=detect_tag_category(tag)
                    )
                    for i, tag in enumerate(raw_tags[:request.parameters.max_tags_per_frame])
                ]

            # Determine scene index
            scene_index = None
            if scene_boundaries:
                ts = meta["timestamp"]
                for i, scene in enumerate(scene_boundaries):
                    if scene["start_timestamp"] <= ts <= scene["end_timestamp"]:
                        scene_index = i
                        break

            frame_results.append(FrameCaption(
                frame_index=meta.get("index", 0),
                timestamp=meta["timestamp"],
                raw_caption=caption,
                tags=aligned_tags,
                summary=summary_data,
                scene_index=scene_index,
                prompt_type_used=request.parameters.prompt_type
            ))

        # Step 8: Aggregate scene summaries
        scene_summaries = None
        if scene_boundaries:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.95,
                stage="aggregating",
                message="Aggregating scene summaries"
            )
            scene_summaries = aggregate_scene_captions(frame_results, scene_boundaries)

        # Build final results
        processing_time = time.time() - start_time
        vram_peak = joycaption.get_vram_usage()[0] if joycaption.model_loaded else None

        result_metadata = CaptionMetadata(
            source=request.source,
            total_frames=frame_result.get("metadata", {}).get("total_frames", 0),
            frames_captioned=len(frame_results),
            model="joycaption",
            model_variant="alpha-two",
            quantization="4-bit" if request.parameters.use_quantization else "none",
            prompt_type=request.parameters.prompt_type,
            processing_time_seconds=processing_time,
            device=CAPTION_DEVICE,
            vram_peak_mb=vram_peak,
            gpu_wait_time_seconds=gpu_wait_time
        )

        outcome = CaptionOutcome(
            frames=frame_results,
            scene_summaries=scene_summaries,
            metadata=result_metadata
        )

        results = {
            "job_id": job_id,
            "source_id": request.source_id,
            "status": JobStatus.COMPLETED.value,
            "captions": outcome.dict(),
            "metadata": result_metadata.dict()
        }

        # Cache results
        await cache_manager.cache_job_results(job_id, cache_key, results)

        # Update final status
        await cache_manager.update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=1.0,
            stage="completed",
            message=f"Captioned {len(frame_results)} frames in {processing_time:.1f}s"
        )

        logger.info(f"Job {job_id} completed in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await cache_manager.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            progress=0.0,
            stage="failed",
            message=f"Job failed: {str(e)[:100]}",
            error=str(e)
        )

    finally:
        # Release GPU if we have a lease
        if resource_client and resource_client.has_gpu_lease:
            try:
                await resource_client.release_gpu()
            except Exception as e:
                logger.warning(f"Failed to release GPU: {e}")

        # Cleanup
        if 'valid_frames' in locals():
            del valid_frames
        if 'frames' in locals():
            del frames
        gc.collect()


def parse_scene_summary_json(caption: str) -> Optional[SceneSummaryData]:
    """
    Parse SCENE_SUMMARY JSON response from JoyCaption VLM.

    The VLM may return JSON wrapped in markdown code blocks or with extra text.
    This function extracts and parses the JSON content.

    Args:
        caption: Raw caption output from JoyCaption

    Returns:
        SceneSummaryData object if parsing successful, None otherwise
    """
    import json
    import re

    if not caption or not caption.strip():
        return None

    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', caption)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'(\{[\s\S]*\})', caption)
            if json_match:
                json_str = json_match.group(1)
            else:
                logger.warning(f"No JSON found in caption: {caption[:100]}...")
                return None

        # Parse JSON
        data = json.loads(json_str)

        # Build nested objects
        cinematography = None
        if data.get("cinematography"):
            cine = data["cinematography"]
            cinematography = CinematographyInfo(
                shot_type=cine.get("shot_type"),
                camera_angle=cine.get("camera_angle"),
                camera_movement=cine.get("camera_movement"),
                focus=cine.get("focus"),
                composition=cine.get("composition"),
                framing=cine.get("framing")
            )

        visual_style = None
        if data.get("visual_style"):
            vs = data["visual_style"]
            visual_style = VisualStyle(
                color_palette=vs.get("color_palette", []),
                color_grading=vs.get("color_grading"),
                contrast=vs.get("contrast"),
                saturation=vs.get("saturation"),
                film_grain=vs.get("film_grain"),
                quality=vs.get("quality"),
                visual_style=vs.get("visual_style"),
                era_aesthetic=vs.get("era_aesthetic")
            )

        environment = None
        if data.get("environment"):
            env = data["environment"]
            environment = EnvironmentInfo(
                time_of_day=env.get("time_of_day"),
                weather=env.get("weather"),
                season=env.get("season"),
                atmosphere=env.get("atmosphere"),
                ambient_light=env.get("ambient_light")
            )

        # Build SceneSummaryData
        return SceneSummaryData(
            locale=data.get("locale"),
            setting=data.get("setting"),
            location_details=data.get("location_details"),
            persons=data.get("persons"),
            attire=data.get("attire", []),
            interactions=data.get("interactions"),
            objects=data.get("objects", []),
            furniture=data.get("furniture", []),
            background_elements=data.get("background_elements", []),
            foreground_elements=data.get("foreground_elements", []),
            text_visible=data.get("text_visible"),
            activities=data.get("activities", []),
            action_intensity=data.get("action_intensity"),
            cinematography=cinematography,
            visual_style=visual_style,
            environment=environment,
            lighting=data.get("lighting"),
            lighting_type=data.get("lighting_type"),
            mood=data.get("mood"),
            tension_level=data.get("tension_level"),
            genre=data.get("genre"),
            sub_genre=data.get("sub_genre"),
            content_type=data.get("content_type"),
            narrative_context=data.get("narrative_context"),
            notable_features=data.get("notable_features", [])
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse SCENE_SUMMARY JSON: {e}")
        logger.debug(f"Raw caption: {caption[:500]}")
        return None
    except Exception as e:
        logger.warning(f"Error building SceneSummaryData: {e}")
        return None


def aggregate_scene_captions(
    frame_results: List[FrameCaption],
    scene_boundaries: List[Dict]
) -> List[SceneCaptionSummary]:
    """Aggregate frame captions into scene summaries"""
    from collections import Counter

    summaries = []

    for i, scene in enumerate(scene_boundaries):
        start = scene["start_timestamp"]
        end = scene["end_timestamp"]

        # Find frames in this scene
        scene_frames = [
            f for f in frame_results
            if start <= f.timestamp <= end
        ]

        if not scene_frames:
            continue

        # Count tag frequencies
        tag_counts = Counter()
        total_conf = 0.0
        total_tags = 0

        for frame in scene_frames:
            for tag in frame.tags:
                tag_counts[tag.tag] += 1
                total_conf += tag.confidence
                total_tags += 1

        dominant_tags = [tag for tag, _ in tag_counts.most_common(10)]
        avg_confidence = total_conf / total_tags if total_tags > 0 else 0.0

        # Combine raw captions
        combined = " | ".join(f.raw_caption[:100] for f in scene_frames[:3])

        summaries.append(SceneCaptionSummary(
            scene_index=i,
            start_timestamp=start,
            end_timestamp=end,
            dominant_tags=dominant_tags,
            frame_count=len(scene_frames),
            avg_confidence=avg_confidence,
            combined_caption=combined[:500] if combined else None
        ))

    return summaries


@app.post("/captions/analyze", response_model=AnalyzeCaptionsResponse, status_code=202)
async def analyze_captions(request: AnalyzeCaptionsRequest, background_tasks: BackgroundTasks):
    """Submit captioning analysis job"""

    if STUB_MODE:
        return AnalyzeCaptionsResponse(
            job_id=f"captions-stub-{int(time.time())}",
            status=JobStatus.NOT_IMPLEMENTED,
            message="Captioning service is running in stub mode",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    try:
        # Validate video exists
        if not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        # Generate cache key
        cache_params = {
            "prompt_type": request.parameters.prompt_type.value,
            "frame_selection": request.parameters.frame_selection.value,
            "min_confidence": request.parameters.min_confidence,
            "max_tags": request.parameters.max_tags_per_frame,
            "align_taxonomy": request.parameters.align_to_taxonomy,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)

        # Check cache
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)
        if cached_job_id:
            logger.info(f"Cache hit for {request.source}: {cached_job_id}")

            if request.job_id and request.job_id != cached_job_id:
                await cache_manager.create_job_alias(request.job_id, cached_job_id)

            return AnalyzeCaptionsResponse(
                job_id=request.job_id or cached_job_id,
                status=JobStatus.COMPLETED,
                message="Results retrieved from cache",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                cache_hit=True
            )

        # Generate new job ID
        job_id = request.job_id or f"captions-{uuid.uuid4()}"

        # Queue background task
        background_tasks.add_task(process_captioning_job, job_id, request)

        logger.info(f"Captioning job {job_id} queued")

        return AnalyzeCaptionsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message="Captioning job queued",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""

    if STUB_MODE:
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": JobStatus.NOT_IMPLEMENTED.value,
                "progress": 0.0,
                "message": "Captioning service is running in stub mode"
            }
        )

    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            created_at=metadata["created_at"],
            started_at=metadata.get("started_at"),
            completed_at=metadata.get("completed_at"),
            error=metadata.get("error"),
            gpu_wait_position=metadata.get("gpu_wait_position")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/jobs/{job_id}/results", response_model=CaptionResults)
async def get_job_results(job_id: str):
    """Get job results"""

    if STUB_MODE:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Captioning service is running in stub mode",
                    "type": "NotImplementedError"
                }
            }
        )

    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (status: {metadata['status']})"
            )

        results = await cache_manager.get_job_results(job_id)

        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return CaptionResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy():
    """Get current tag taxonomy"""

    if not tag_aligner:
        return TaxonomyResponse(
            tags=[],
            categories=[],
            total_tags=0,
            last_synced=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        )

    stats = tag_aligner.get_taxonomy_stats()

    return TaxonomyResponse(
        tags=tag_aligner.taxonomy,
        categories=list(tag_aligner.category_lookup.keys()),
        total_tags=stats["total_tags"],
        last_synced=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    )


@app.post("/captions/taxonomy/sync")
async def sync_taxonomy():
    """Sync taxonomy from Stash"""
    global tag_aligner, hierarchical_tagger

    if STUB_MODE:
        return {"status": "stub_mode", "message": "Running in stub mode"}

    try:
        if not await stash_client.health_check():
            raise HTTPException(status_code=503, detail="Stash not available")

        taxonomy = await stash_client.get_all_tags()
        tag_aligner = TagAligner(taxonomy)
        hierarchical_tagger = HierarchicalTagger(taxonomy)

        return {
            "status": "synced",
            "tags_loaded": len(taxonomy),
            "hierarchy_stats": hierarchical_tagger.get_hierarchy_stats(),
            "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing taxonomy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/captions/taxonomy/upload")
async def upload_taxonomy(tags: List[TagTaxonomyNode]):
    """
    Upload taxonomy directly via API request.

    Use this endpoint when Stash is not available or for standalone operation.
    The uploaded taxonomy will replace any existing taxonomy.

    Example request body:
    ```json
    [
        {"id": "1", "name": "Indoor", "aliases": ["indoors"], "parent_id": null, "children": ["2", "3"]},
        {"id": "2", "name": "Bedroom", "aliases": [], "parent_id": "1", "children": []},
        {"id": "3", "name": "Office", "aliases": ["workspace"], "parent_id": "1", "children": []}
    ]
    ```
    """
    global tag_aligner, hierarchical_tagger

    if STUB_MODE:
        return {"status": "stub_mode", "message": "Running in stub mode"}

    try:
        tag_aligner = TagAligner(tags)
        hierarchical_tagger = HierarchicalTagger(tags)

        logger.info(f"Taxonomy uploaded: {len(tags)} tags")

        return {
            "status": "uploaded",
            "tags_loaded": len(tags),
            "hierarchy_stats": hierarchical_tagger.get_hierarchy_stats(),
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }

    except Exception as e:
        logger.error(f"Error uploading taxonomy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    if STUB_MODE:
        return HealthResponse(
            status="healthy",
            implemented=False,
            message="Stub service - JoyCaption not loaded",
            model=None,
            model_loaded=False,
            device=None,
            gpu_acquired=False
        )

    model_info = joycaption.get_model_info() if joycaption else {}

    return HealthResponse(
        status="healthy",
        implemented=True,
        message="JoyCaption VLM service active",
        model=model_info.get("name"),
        model_loaded=model_info.get("loaded", False),
        device=model_info.get("device"),
        vram_available_mb=model_info.get("vram_mb"),
        gpu_acquired=resource_client.has_gpu_lease if resource_client else False,
        default_min_confidence=float(os.getenv("CAPTIONS_MIN_CONFIDENCE", "0.5"))
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)
