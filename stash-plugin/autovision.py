#!/usr/bin/env python3
"""
Stash Auto Vision Plugin - Scene Captioning
Triggers JoyCaption VLM captioning and applies tags to scenes
"""

import json
import sys
import time
import requests
from typing import Optional, Dict, Any, List


def log(level: str, message: str):
    """Log message to Stash plugin output"""
    print(json.dumps({"level": level, "message": message}))


def progress(current: float, message: str = ""):
    """Report progress to Stash"""
    print(json.dumps({"progress": current, "message": message}))


def get_stash_connection() -> Dict[str, str]:
    """Get Stash connection info from plugin context"""
    # Read from stdin (Stash provides this)
    try:
        input_data = json.loads(sys.stdin.read())
        server = input_data.get("server_connection", {})
        return {
            "url": server.get("Scheme", "http") + "://" + server.get("Host", "localhost:9999"),
            "api_key": server.get("SessionCookie", {}).get("Value", "")
        }
    except Exception as e:
        log("error", f"Failed to get Stash connection: {e}")
        return {"url": "http://localhost:9999", "api_key": ""}


def get_plugin_config(stash: Dict[str, str]) -> Dict[str, Any]:
    """Get plugin configuration from Stash"""
    query = """
    query Configuration {
        configuration {
            plugins
        }
    }
    """

    try:
        response = requests.post(
            f"{stash['url']}/graphql",
            json={"query": query},
            headers={
                "Content-Type": "application/json",
                "Cookie": f"session={stash['api_key']}" if stash.get("api_key") else ""
            }
        )
        data = response.json()
        plugins_config = data.get("data", {}).get("configuration", {}).get("plugins", {})
        return plugins_config.get("autovision", {})
    except Exception as e:
        log("warning", f"Failed to get plugin config: {e}")
        return {}


def get_scene_info(stash: Dict[str, str], scene_id: str) -> Optional[Dict[str, Any]]:
    """Get scene information from Stash"""
    query = """
    query FindScene($id: ID!) {
        findScene(id: $id) {
            id
            title
            path
            paths {
                screenshot
                sprite
                vtt
            }
            tags {
                id
                name
            }
        }
    }
    """

    try:
        response = requests.post(
            f"{stash['url']}/graphql",
            json={"query": query, "variables": {"id": scene_id}},
            headers={
                "Content-Type": "application/json",
                "Cookie": f"session={stash['api_key']}" if stash.get("api_key") else ""
            }
        )
        data = response.json()
        return data.get("data", {}).get("findScene")
    except Exception as e:
        log("error", f"Failed to get scene info: {e}")
        return None


def apply_tags_to_scene(
    stash: Dict[str, str],
    scene_id: str,
    tag_ids: List[str],
    existing_tag_ids: List[str]
) -> bool:
    """Apply tags to a scene in Stash"""
    # Merge existing and new tags
    all_tag_ids = list(set(existing_tag_ids + tag_ids))

    mutation = """
    mutation SceneUpdate($input: SceneUpdateInput!) {
        sceneUpdate(input: $input) {
            id
            tags {
                id
                name
            }
        }
    }
    """

    try:
        response = requests.post(
            f"{stash['url']}/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "id": scene_id,
                        "tag_ids": all_tag_ids
                    }
                }
            },
            headers={
                "Content-Type": "application/json",
                "Cookie": f"session={stash['api_key']}" if stash.get("api_key") else ""
            }
        )
        data = response.json()
        if "errors" in data:
            log("error", f"GraphQL errors: {data['errors']}")
            return False
        return True
    except Exception as e:
        log("error", f"Failed to apply tags: {e}")
        return False


def submit_captioning_job(
    api_url: str,
    video_path: str,
    source_id: str,
    config: Dict[str, Any],
    sprite_vtt_url: Optional[str] = None,
    sprite_image_url: Optional[str] = None
) -> Optional[str]:
    """Submit a captioning job to the captioning service"""
    request_body = {
        "source": video_path,
        "source_id": source_id,
        "parameters": {
            "prompt_type": config.get("prompt_type", "scene_summary"),
            "min_confidence": config.get("min_confidence", 0.5),
            "max_tags_per_frame": config.get("max_tags_per_scene", 20),
            "align_to_taxonomy": True,
            "use_hierarchical_scoring": config.get("use_hierarchical_scoring", True),
            "select_sharpest": True
        }
    }

    # Use sprite sheets if available and enabled
    if config.get("use_sprite_sheets") and sprite_vtt_url and sprite_image_url:
        request_body["parameters"]["frame_selection"] = "sprite_sheet"
        request_body["parameters"]["sprite_vtt_url"] = sprite_vtt_url
        request_body["parameters"]["sprite_image_url"] = sprite_image_url

    try:
        response = requests.post(
            f"{api_url}/captions/analyze",
            json=request_body,
            timeout=30
        )
        data = response.json()

        if response.status_code in (200, 202):
            return data.get("job_id")
        else:
            log("error", f"Captioning request failed: {data}")
            return None
    except Exception as e:
        log("error", f"Failed to submit captioning job: {e}")
        return None


def poll_job_status(api_url: str, job_id: str, timeout: float = 600.0) -> Optional[Dict[str, Any]]:
    """Poll for job completion"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{api_url}/captions/jobs/{job_id}/status", timeout=10)
            data = response.json()

            status = data.get("status")
            prog = data.get("progress", 0.0)
            stage = data.get("stage", "")

            progress(prog, f"{stage}: {data.get('message', '')}")

            if status == "completed":
                return data
            elif status == "failed":
                log("error", f"Job failed: {data.get('error', 'Unknown error')}")
                return None

            time.sleep(2)
        except Exception as e:
            log("warning", f"Error polling job status: {e}")
            time.sleep(5)

    log("error", f"Job timed out after {timeout}s")
    return None


def get_job_results(api_url: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Get job results"""
    try:
        response = requests.get(f"{api_url}/captions/jobs/{job_id}/results", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            log("error", f"Failed to get results: {response.text}")
            return None
    except Exception as e:
        log("error", f"Error getting job results: {e}")
        return None


def sync_taxonomy(api_url: str) -> bool:
    """Trigger taxonomy sync on captioning service"""
    try:
        response = requests.post(f"{api_url}/captions/taxonomy/sync", timeout=30)
        if response.status_code == 200:
            data = response.json()
            log("info", f"Taxonomy synced: {data.get('tags_loaded', 0)} tags")
            return True
        else:
            log("error", f"Taxonomy sync failed: {response.text}")
            return False
    except Exception as e:
        log("error", f"Error syncing taxonomy: {e}")
        return False


def caption_scene(scene_id: str):
    """Caption a single scene and apply tags"""
    stash = get_stash_connection()
    config = get_plugin_config(stash)
    api_url = config.get("api_url", "http://localhost:5006")

    progress(0.0, "Getting scene information...")

    scene = get_scene_info(stash, scene_id)
    if not scene:
        log("error", f"Scene not found: {scene_id}")
        return

    video_path = scene.get("path")
    if not video_path:
        log("error", "Scene has no video path")
        return

    log("info", f"Captioning scene: {scene.get('title', video_path)}")

    # Get sprite sheet URLs if available
    paths = scene.get("paths", {})
    sprite_vtt_url = paths.get("vtt")
    sprite_image_url = paths.get("sprite")

    progress(0.1, "Submitting captioning job...")

    # Submit job
    job_id = submit_captioning_job(
        api_url=api_url,
        video_path=video_path,
        source_id=f"stash-scene-{scene_id}",
        config=config,
        sprite_vtt_url=sprite_vtt_url,
        sprite_image_url=sprite_image_url
    )

    if not job_id:
        log("error", "Failed to submit captioning job")
        return

    log("info", f"Job submitted: {job_id}")

    # Poll for completion
    result = poll_job_status(api_url, job_id)
    if not result:
        return

    progress(0.9, "Retrieving results...")

    # Get results
    results = get_job_results(api_url, job_id)
    if not results:
        return

    # Extract tag IDs from results
    tag_ids = []
    captions = results.get("captions", {})
    frames = captions.get("frames", [])

    for frame in frames:
        for tag in frame.get("tags", []):
            if tag.get("stash_tag_id"):
                tag_ids.append(tag["stash_tag_id"])

    # Deduplicate and limit
    tag_ids = list(set(tag_ids))[:config.get("max_tags_per_scene", 20)]

    if config.get("apply_tags", True) and tag_ids:
        progress(0.95, f"Applying {len(tag_ids)} tags...")

        existing_tag_ids = [t["id"] for t in scene.get("tags", [])]
        success = apply_tags_to_scene(stash, scene_id, tag_ids, existing_tag_ids)

        if success:
            log("info", f"Applied {len(tag_ids)} tags to scene")
        else:
            log("warning", "Failed to apply some tags")

    progress(1.0, "Complete")
    log("info", "Captioning complete")


def handle_scene_created(scene_id: str):
    """Hook handler for scene creation"""
    stash = get_stash_connection()
    config = get_plugin_config(stash)

    if not config.get("auto_caption_enabled", False):
        log("debug", "Auto-captioning disabled, skipping")
        return

    log("info", f"Auto-captioning new scene: {scene_id}")
    caption_scene(scene_id)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        log("error", "No command specified")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "task":
        if len(sys.argv) < 3:
            log("error", "No task specified")
            sys.exit(1)

        task = sys.argv[2]

        if task == "caption_scene":
            # Get scene ID from input
            try:
                input_data = json.loads(sys.stdin.read())
                scene_ids = input_data.get("args", {}).get("scene_ids", [])
                if scene_ids:
                    for scene_id in scene_ids:
                        caption_scene(scene_id)
                else:
                    log("error", "No scene ID provided")
            except Exception as e:
                log("error", f"Failed to parse input: {e}")

        elif task == "caption_selected":
            try:
                input_data = json.loads(sys.stdin.read())
                scene_ids = input_data.get("args", {}).get("scene_ids", [])
                total = len(scene_ids)
                for i, scene_id in enumerate(scene_ids):
                    progress(i / total, f"Captioning scene {i + 1}/{total}")
                    caption_scene(scene_id)
                progress(1.0, f"Captioned {total} scenes")
            except Exception as e:
                log("error", f"Failed to caption scenes: {e}")

        elif task == "sync_taxonomy":
            stash = get_stash_connection()
            config = get_plugin_config(stash)
            api_url = config.get("api_url", "http://localhost:5006")
            sync_taxonomy(api_url)

        else:
            log("error", f"Unknown task: {task}")

    elif mode == "hook":
        if len(sys.argv) < 3:
            log("error", "No hook specified")
            sys.exit(1)

        hook = sys.argv[2]

        if hook == "scene_created":
            try:
                input_data = json.loads(sys.stdin.read())
                scene_id = input_data.get("args", {}).get("hookContext", {}).get("id")
                if scene_id:
                    handle_scene_created(scene_id)
                else:
                    log("warning", "No scene ID in hook context")
            except Exception as e:
                log("error", f"Hook error: {e}")
        else:
            log("error", f"Unknown hook: {hook}")

    else:
        log("error", f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
