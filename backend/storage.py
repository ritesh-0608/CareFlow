
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent / ".env")


BUCKET_NAME = "careflow-datasets"


@lru_cache(maxsize=1)
def get_storage_client() -> Client:
    """Create and cache the Supabase client when cloud storage is configured."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Supabase Storage is not configured. "
            "Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )

    return create_client(supabase_url, supabase_key)


def upload_dataset(storage_path: str, file_bytes: bytes) -> None:
    """Upload a dataset to the private Supabase Storage bucket."""
    get_storage_client().storage.from_(BUCKET_NAME).upload(
        path=storage_path,
        file=file_bytes,
        file_options={
            "content-type": "text/csv",
            "upsert": "false",
        },
    )


def download_dataset(storage_path: str) -> bytes:
    """Download a dataset from private Supabase Storage."""
    return get_storage_client().storage.from_(BUCKET_NAME).download(
        storage_path
    )


def delete_dataset(storage_path: str) -> None:
    """Delete a dataset from private Supabase Storage."""
    get_storage_client().storage.from_(BUCKET_NAME).remove(
        [storage_path]
    )
