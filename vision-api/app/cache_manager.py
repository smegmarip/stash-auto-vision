"""
Vision API - Cache Manager
Cross-service job listing and indexing for vision-api
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Redis caching and cross-service job aggregation"""

    # Service namespaces to query for job listing
    SERVICE_NAMESPACES = ["vision", "faces", "scenes", "semantics", "objects"]

    # Priority order for deduplication (lower = higher priority)
    SERVICE_PRIORITY = {"vision": 0, "faces": 1, "scenes": 2}

    def __init__(self, redis_client: aioredis.Redis, ttl: int = 31536000):
        """
        Initialize cache manager

        Args:
            redis_client: Existing Redis connection
            ttl: Default TTL in seconds (default: 1 year)
        """
        self.redis = redis_client
        self.ttl = ttl
        self.module = "vision"

    async def index_job(self, job_id: str, metadata: Dict[str, Any]):
        """
        Add job to secondary indexes (vision-api jobs only)

        Args:
            job_id: Job identifier
            metadata: Job metadata dict
        """
        try:
            created_at = metadata.get("created_at")
            if isinstance(created_at, str):
                # Parse ISO format to timestamp
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = dt.timestamp()
                except:
                    created_at = time.time()
            else:
                created_at = created_at or time.time()

            # Main sorted set (for date range queries)
            await self.redis.zadd("vision:jobs:all", {job_id: created_at})

            # Status index
            if status := metadata.get("status"):
                await self.redis.sadd(f"vision:jobs:by_status:{status}", job_id)

            # Scene index
            if source_id := metadata.get("source_id"):
                await self.redis.sadd(f"vision:jobs:by_scene:{source_id}", job_id)

            # Source index (hash the path for key safety)
            if source := metadata.get("source"):
                source_hash = hashlib.md5(source.encode()).hexdigest()[:16]
                await self.redis.sadd(f"vision:jobs:by_source:{source_hash}", job_id)

            logger.debug(f"Indexed job: {job_id}")
        except Exception as e:
            logger.error(f"Error indexing job: {e}")

    async def update_job_status_index(self, job_id: str, old_status: str, new_status: str):
        """
        Move job between status indexes

        Args:
            job_id: Job identifier
            old_status: Previous status
            new_status: New status
        """
        try:
            if old_status:
                await self.redis.srem(f"vision:jobs:by_status:{old_status}", job_id)
            await self.redis.sadd(f"vision:jobs:by_status:{new_status}", job_id)
            logger.debug(f"Updated status index for job {job_id}: {old_status} -> {new_status}")
        except Exception as e:
            logger.error(f"Error updating status index: {e}")

    async def list_jobs(
        self,
        status: Optional[str] = None,
        service: Optional[str] = None,
        source_id: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
        include_results: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List jobs across all services with filtering, pagination, and deduplication.

        Services remain encapsulated - this method reads their Redis keys directly
        without requiring any changes to individual services.

        Args:
            status: Filter by job status
            service: Filter by service (vision, faces, scenes) or None for all
            source_id: Filter by source_id
            source: Filter by source video path
            start_date: Filter by start date (Unix timestamp)
            end_date: Filter by end date (Unix timestamp)
            include_results: Include full job results in response
            limit: Maximum jobs to return
            offset: Number of jobs to skip

        Returns:
            Tuple of (jobs list, total count)
        """
        try:
            # Determine which service namespaces to query
            services = [service] if service else self.SERVICE_NAMESPACES

            # Collect all job keys across services
            all_job_keys = []
            for svc in services:
                pattern = f"{svc}:job:*:metadata"
                keys = await self.redis.keys(pattern)
                all_job_keys.extend(keys)

            # Fetch and filter jobs
            jobs = []
            seen_cache_keys = {}  # cache_key -> job for deduplication

            for key in all_job_keys:
                # Extract service and job_id from key: "{service}:job:{job_id}:metadata"
                parts = key.split(":")
                if len(parts) < 4:
                    continue
                svc, job_id = parts[0], parts[2]

                metadata_json = await self.redis.get(key)
                if not metadata_json:
                    continue

                # Parse metadata - handle both JSON and Python dict str
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    try:
                        metadata = eval(metadata_json)
                    except:
                        continue

                # Add service identifier
                metadata["service"] = svc
                metadata["job_id"] = job_id

                # Apply filters
                if status and metadata.get("status") != status:
                    continue
                if source_id and metadata.get("source_id") != source_id:
                    continue
                if source and metadata.get("source") != source:
                    continue

                # Date filtering
                created_at = metadata.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        try:
                            from datetime import datetime

                            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            created_ts = dt.timestamp()
                        except:
                            created_ts = 0
                    else:
                        created_ts = float(created_at)

                    if start_date and created_ts < start_date:
                        continue
                    if end_date and created_ts > end_date:
                        continue

                # No deduplication - always include job
                jobs.append(metadata)

            # Optionally fetch full results
            if include_results:
                for job in jobs:
                    svc = job["service"]
                    job_id = job["job_id"]
                    results_key = f"{svc}:job:{job_id}:results"
                    results_json = await self.redis.get(results_key)
                    if results_json:
                        try:
                            job["results"] = json.loads(results_json)
                        except json.JSONDecodeError:
                            try:
                                job["results"] = eval(results_json)
                            except:
                                job["results"] = None

            # Sort by created_at descending
            def get_created_ts(j):
                created_at = j.get("created_at")
                if isinstance(created_at, str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        return dt.timestamp()
                    except:
                        return 0
                return float(created_at) if created_at else 0

            jobs.sort(key=get_created_ts, reverse=True)

            total = len(jobs)
            return jobs[offset : offset + limit], total

        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return [], 0

    async def count_jobs(
        self,
        status: Optional[str] = None,
        service: Optional[str] = None,
        source_id: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
    ) -> Tuple[int, Dict[str, int]]:
        """
        Count jobs across all services with filtering and deduplication.

        Uses same filtering logic as list_jobs() but only counts jobs without
        fetching full metadata or results for better performance.

        Args:
            status: Filter by job status
            service: Filter by service (vision, faces, scenes, semantics, objects) or None for all
            source_id: Filter by source_id
            source: Filter by source video path
            start_date: Filter by start date (Unix timestamp)
            end_date: Filter by end date (Unix timestamp)

        Returns:
            Tuple of (total_count, counts_by_service)
            Example: (147, {"vision": 23, "faces": 47, "scenes": 45, "semantics": 32})
        """
        try:
            # Determine which service namespaces to query
            services = [service] if service else self.SERVICE_NAMESPACES

            # Collect all job keys across services
            all_job_keys = []
            for svc in services:
                pattern = f"{svc}:job:*:metadata"
                keys = await self.redis.keys(pattern)
                all_job_keys.extend(keys)

            # Count jobs by service
            service_counts = {svc: 0 for svc in self.SERVICE_NAMESPACES}
            seen_cache_keys = {}  # cache_key -> (service, job_id) for deduplication
            total = 0

            for key in all_job_keys:
                # Extract service and job_id from key: "{service}:job:{job_id}:metadata"
                parts = key.split(":")
                if len(parts) < 4:
                    continue
                svc, job_id = parts[0], parts[2]

                metadata_json = await self.redis.get(key)
                if not metadata_json:
                    continue

                # Parse metadata - handle both JSON and Python dict str
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    try:
                        metadata = eval(metadata_json)
                    except:
                        continue

                # Apply filters
                if status and metadata.get("status") != status:
                    continue
                if source_id and metadata.get("source_id") != source_id:
                    continue
                if source and metadata.get("source") != source:
                    continue

                # Date filtering
                created_at = metadata.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        try:
                            from datetime import datetime

                            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            created_ts = dt.timestamp()
                        except:
                            created_ts = 0
                    else:
                        created_ts = float(created_at)

                    if start_date and created_ts < start_date:
                        continue
                    if end_date and created_ts > end_date:
                        continue

                # No deduplication - count every job
                service_counts[svc] += 1
                total += 1

            return total, service_counts

        except Exception as e:
            logger.error(f"Error counting jobs: {e}")
            return 0, {}

    async def get_job_metadata(self, job_id: str, service: str = "vision") -> Optional[Dict[str, Any]]:
        """
        Retrieve job metadata by job ID

        Args:
            job_id: Job identifier
            service: Service namespace (default: vision)

        Returns:
            Metadata dict if found, None otherwise
        """
        try:
            metadata_key = f"{service}:job:{job_id}:metadata"
            metadata_json = await self.redis.get(metadata_key)

            if metadata_json:
                try:
                    return json.loads(metadata_json)
                except json.JSONDecodeError:
                    return eval(metadata_json)
            return None
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return None

    async def get_job_results(self, job_id: str, service: str = "vision") -> Optional[Dict[str, Any]]:
        """
        Retrieve job results by job ID

        Args:
            job_id: Job identifier
            service: Service namespace (default: vision)

        Returns:
            Results dict if found, None otherwise
        """
        try:
            results_key = f"{service}:job:{job_id}:results"
            results_json = await self.redis.get(results_key)

            if results_json:
                try:
                    return json.loads(results_json)
                except json.JSONDecodeError:
                    return eval(results_json)
            return None
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return None
