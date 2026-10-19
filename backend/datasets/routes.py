from datetime import date
from pathlib import Path
from uuid import uuid4

from fastapi.responses import FileResponse, Response

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, get_db

from backend.database.models import (
    AuditLog,
    Dataset,
    Forecast,
    ResourceRecord,
    TrainingRun,
    User,
)

from backend.datasets.persistence import (
    save_forecast,
    save_resource_records,
    save_training_run,
)

from backend.datasets.processor import (
    forecast_hospital,
    forecast_resource,
    read_uploaded_csv,
)
from backend.storage import (
    upload_dataset as upload_dataset_to_storage,
    download_dataset as download_dataset_from_storage,
    delete_dataset as delete_dataset_from_storage,
)

from backend.datasets.validator import (
    DatasetValidationError,
    validate_dataset,
    validate_file_size,
    validate_filename,
    validate_row_count_limit,
)

from backend.forecasting.disease import (
    forecast_all_diseases,
)


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent.parent / "uploads"
)


ALLOWED_DATASET_TYPES = {
    "hospital_history",
    "hospital_resources",
    "disease_data",
}


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # 1. Validate dataset type
    # --------------------------------------------------------
    if dataset_type not in ALLOWED_DATASET_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid dataset_type. Allowed values are "
                "'hospital_history', 'hospital_resources', "
                "and 'disease_data'."
            ),
        )

    # --------------------------------------------------------
    # 2. Validate filename
    # --------------------------------------------------------
    try:
        safe_filename = validate_filename(
            file.filename or ""
        )

    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # 3. Read uploaded file
    # --------------------------------------------------------
    file_content = await file.read()

    # --------------------------------------------------------
    # 4. Validate file size
    # --------------------------------------------------------
    try:
        validate_file_size(
            len(file_content)
        )

    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # 5. Prepare local upload directory
    # --------------------------------------------------------
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 6. Generate unique stored filename
    # --------------------------------------------------------
    stored_filename = (
        f"{uuid4().hex}_{safe_filename}"
    )

    file_path = UPLOAD_DIR / stored_filename

    # --------------------------------------------------------
    # 7. Save and validate CSV
    # --------------------------------------------------------
    try:
        file_path.write_bytes(
            file_content
        )

        validation_result = validate_dataset(
            file_path,
            dataset_type,
        )

        validate_row_count_limit(
            validation_result["row_count"]
        )

    except DatasetValidationError as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset validation failed due to an unexpected file-processing error.",
        ) from exc

    # --------------------------------------------------------
    # 8. Reject invalid dataset
    # --------------------------------------------------------
    if not validation_result["valid"]:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Dataset validation failed.",
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"],
            },
        )

    # Upload the validated CSV to persistent Supabase Storage.
    storage_path = f"{current_user.id}/{stored_filename}"

    try:
        upload_dataset_to_storage(storage_path, file_content)
    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to store dataset in persistent storage.",
        ) from exc

    # --------------------------------------------------------
    # 9. Create dataset version and update current dataset
    # --------------------------------------------------------
    try:
        # Find the current dataset for this user and dataset type.
        current_dataset = (
            db.query(Dataset)
            .filter(
                Dataset.user_id == current_user.id,
                Dataset.dataset_type == dataset_type,
                Dataset.is_current.is_(True),
            )
            .first()
        )

        # Find the highest version number already used for this
        # user and dataset type. Version numbers are never reused.
        latest_dataset = (
            db.query(Dataset)
            .filter(
                Dataset.user_id == current_user.id,
                Dataset.dataset_type == dataset_type,
            )
            .order_by(Dataset.version_number.desc())
            .first()
        )

        if latest_dataset:
            next_version = latest_dataset.version_number + 1
        else:
            next_version = 1

        # Existing current dataset becomes a previous version.
        if current_dataset:
            current_dataset.is_current = False

        # Create the new dataset as the current version.
        dataset = Dataset(
            user_id=current_user.id,
            filename=safe_filename,
            dataset_type=dataset_type,
            start_date=date.fromisoformat(
                validation_result["date_validation"][
                    "start_date"
                ]
            ),
            end_date=date.fromisoformat(
                validation_result["date_validation"][
                    "end_date"
                ]
            ),
            status="uploaded",
            is_current=True,
            version_number=next_version,
            file_path=f"supabase://{storage_path}",
        )

        db.add(dataset)

        # Flush so the new dataset receives its database ID
        # before creating the audit record.
        db.flush()

        # Record the dataset-version creation in the audit trail.
        audit_log = AuditLog(
            user_id=current_user.id,
            action="DATASET_VERSION_CREATED",
            details={
                "dataset_id": dataset.id,
                "dataset_type": dataset_type,
                "version_number": next_version,
                "previous_dataset_id": (
                    current_dataset.id
                    if current_dataset
                    else None
                ),
            },
        )

        db.add(audit_log)

        # Commit the complete version transition and its audit record:
        # old current -> previous
        # new dataset -> current
        # audit record -> DATASET_VERSION_CREATED
        db.commit()
        db.refresh(dataset)

    except Exception as exc:
        db.rollback()

        if file_path.exists():
            file_path.unlink()
        try:
            delete_dataset_from_storage(storage_path)
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="Failed to create dataset version due to a database or server error.",
        ) from exc

    # --------------------------------------------------------
    # 10. Process hospital history dataset
    # --------------------------------------------------------
    if dataset_type == "hospital_history":
        try:
            dataframe = read_uploaded_csv(
                str(file_path)
            )

            # ----------------------------------------------------
            # Admissions forecast
            # ----------------------------------------------------
            admissions_result = forecast_hospital(
                dataframe,
                "admissions",
                7,
            )

            admissions_training_run = save_training_run(
                db=db,
                dataset_id=dataset.id,
                training_period_start=date.fromisoformat(
                    admissions_result[
                        "training_period_start"
                    ]
                ),
                training_period_end=date.fromisoformat(
                    admissions_result[
                        "training_period_end"
                    ]
                ),
                model_name=admissions_result[
                    "model_name"
                ],
                model_version=admissions_result[
                    "model_version"
                ],
                feature_schema_version="hospital-features-v1",
                validation_metrics=admissions_result[
                    "validation_metrics"
                ],
                test_metrics=admissions_result[
                    "test_metrics"
                ],
            )

            admissions_forecast = save_forecast(
                db=db,
                user_id=current_user.id,
                dataset_id=dataset.id,
                training_run_id=admissions_training_run.id,
                forecast_type="admissions",
                horizon=7,
                model_name=admissions_result[
                    "model_name"
                ],
                model_version=admissions_result[
                    "model_version"
                ],
                results={
                    "predictions": admissions_result[
                        "predictions"
                    ]
                },
                performance_metrics={
                    "validation": admissions_result[
                        "validation_metrics"
                    ],
                    "test": admissions_result[
                        "test_metrics"
                    ],
                },
            )

            # ----------------------------------------------------
            # ICU demand forecast
            # ----------------------------------------------------
            icu_result = forecast_hospital(
                dataframe,
                "icu_demand",
                7,
            )

            icu_training_run = save_training_run(
                db=db,
                dataset_id=dataset.id,
                training_period_start=date.fromisoformat(
                    icu_result[
                        "training_period_start"
                    ]
                ),
                training_period_end=date.fromisoformat(
                    icu_result[
                        "training_period_end"
                    ]
                ),
                model_name=icu_result[
                    "model_name"
                ],
                model_version=icu_result[
                    "model_version"
                ],
                feature_schema_version="hospital-features-v1",
                validation_metrics=icu_result[
                    "validation_metrics"
                ],
                test_metrics=icu_result[
                    "test_metrics"
                ],
            )

            icu_forecast = save_forecast(
                db=db,
                user_id=current_user.id,
                dataset_id=dataset.id,
                training_run_id=icu_training_run.id,
                forecast_type="icu",
                horizon=7,
                model_name=icu_result[
                    "model_name"
                ],
                model_version=icu_result[
                    "model_version"
                ],
                results={
                    "predictions": icu_result[
                        "predictions"
                    ]
                },
                performance_metrics={
                    "validation": icu_result[
                        "validation_metrics"
                    ],
                    "test": icu_result[
                        "test_metrics"
                    ],
                },
            )

            # ----------------------------------------------------
            # Mark dataset as successfully processed
            # ----------------------------------------------------
            dataset.status = "processed"

            db.commit()
            db.refresh(dataset)

        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Hospital dataset processing failed due to a server or forecasting error.",
            )from exc

        # ----------------------------------------------------
        # 11. Return hospital processing response
        # ----------------------------------------------------
        return {
            "message": (
                "Hospital history dataset uploaded, "
                "validated, processed, and forecasts "
                "generated successfully."
            ),
            "dataset_id": dataset.id,
            "filename": safe_filename,
            "dataset_type": dataset_type,
            "row_count": validation_result["row_count"],
            "start_date": validation_result[
                "date_validation"
            ]["start_date"],
            "end_date": validation_result[
                "date_validation"
            ]["end_date"],
            "historical_days": validation_result[
                "historical_period"
            ]["days"],
            "disclaimer": validation_result[
                "historical_period"
            ]["disclaimer"],
            "forecasts": {
                "admissions": {
                    "training_run_id": admissions_training_run.id,
                    "forecast_id": admissions_forecast.id,
                    "forecast_type": admissions_forecast.forecast_type,
                    "horizon": admissions_forecast.horizon,
                    "predictions": admissions_result[
                        "predictions"
                    ],
                    "validation_metrics": admissions_result[
                        "validation_metrics"
                    ],
                    "test_metrics": admissions_result[
                        "test_metrics"
                    ],
                },
                "icu": {
                    "training_run_id": icu_training_run.id,
                    "forecast_id": icu_forecast.id,
                    "forecast_type": icu_forecast.forecast_type,
                    "horizon": icu_forecast.horizon,
                    "predictions": icu_result[
                        "predictions"
                    ],
                    "validation_metrics": icu_result[
                        "validation_metrics"
                    ],
                    "test_metrics": icu_result[
                        "test_metrics"
                    ],
                },
            },
            "user_id": current_user.id,
        }

    # --------------------------------------------------------
    # 12. Process hospital resources dataset
    # --------------------------------------------------------
    if dataset_type == "hospital_resources":
        try:
            dataframe = read_uploaded_csv(
                str(file_path)
            )

            resource_records = save_resource_records(
                db=db,
                dataset_id=dataset.id,
                dataframe=dataframe,
            )

            resource_names = sorted(
                dataframe["resource_name"]
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            resource_results = []

            for resource_name in resource_names:
                forecast_result = forecast_resource(
                    dataframe,
                    resource_name,
                    7,
                )

                training_run = save_training_run(
                    db=db,
                    dataset_id=dataset.id,
                    training_period_start=date.fromisoformat(
                        forecast_result[
                            "training_period_start"
                        ]
                    ),
                    training_period_end=date.fromisoformat(
                        forecast_result[
                            "training_period_end"
                        ]
                    ),
                    model_name=forecast_result[
                        "model_name"
                    ],
                    model_version=forecast_result[
                        "model_version"
                    ],
                    feature_schema_version="resource-features-v1",
                    validation_metrics=forecast_result[
                        "validation_metrics"
                    ],
                    test_metrics=forecast_result[
                        "test_metrics"
                    ],
                )

                forecast = save_forecast(
                    db=db,
                    user_id=current_user.id,
                    dataset_id=dataset.id,
                    training_run_id=training_run.id,
                    forecast_type="resource",
                    horizon=7,
                    model_name=forecast_result[
                        "model_name"
                    ],
                    model_version=forecast_result[
                        "model_version"
                    ],
                    results={
                        "resource_name": resource_name,
                        "predictions": forecast_result[
                            "predictions"
                        ],
                    },
                    performance_metrics={
                        "validation": forecast_result[
                            "validation_metrics"
                        ],
                        "test": forecast_result[
                            "test_metrics"
                        ],
                    },
                )

                resource_results.append(
                    {
                        "resource_name": resource_name,
                        "training_run_id": training_run.id,
                        "forecast_id": forecast.id,
                        "horizon": forecast.horizon,
                        "predictions": forecast_result[
                            "predictions"
                        ],
                        "validation_metrics": forecast_result[
                            "validation_metrics"
                        ],
                        "test_metrics": forecast_result[
                            "test_metrics"
                        ],
                    }
                )

            # ----------------------------------------------------
            # Mark dataset as successfully processed
            # ----------------------------------------------------
            dataset.status = "processed"

            db.commit()
            db.refresh(dataset)

        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Resource dataset processing failed due to a server or forecasting error.",
            ) from exc

        # ----------------------------------------------------
        # 13. Return resource processing response
        # ----------------------------------------------------
        return {
            "message": (
                "Hospital resource dataset uploaded, "
                "validated, processed, and forecast successfully."
            ),
            "dataset_id": dataset.id,
            "filename": safe_filename,
            "dataset_type": dataset_type,
            "row_count": validation_result["row_count"],
            "resource_record_count": len(resource_records),
            "resource_count": len(resource_names),
            "resources": resource_results,
            "start_date": validation_result[
                "date_validation"
            ]["start_date"],
            "end_date": validation_result[
                "date_validation"
            ]["end_date"],
            "historical_days": validation_result[
                "historical_period"
            ]["days"],
            "disclaimer": validation_result[
                "historical_period"
            ]["disclaimer"],
            "user_id": current_user.id,
        }

    # --------------------------------------------------------
    # 14. Process hospital disease dataset
    # --------------------------------------------------------
    if dataset_type == "disease_data":
        try:
            dataframe = read_uploaded_csv(
                str(file_path)
            )

            disease_processing_result = forecast_all_diseases(
                dataframe,
                7,
            )

            disease_results = []

            for disease_result in disease_processing_result[
                "diseases"
            ]:
                disease_name = disease_result["disease"]

                # ------------------------------------------------
                # Limited-history diseases are returned with their
                # warning/disclaimer but do not create a
                # TrainingRun or Forecast because no model/forecast
                # was actually generated.
                # ------------------------------------------------
                if not disease_result["available"]:
                    disease_results.append(
                        {
                            "disease": disease_name,
                            "status": disease_result["status"],
                            "available": False,
                            "horizon": disease_result["horizon"],
                            "predictions": [],
                            "reason": disease_result.get(
                                "reason"
                            ),
                            "history": disease_result.get(
                                "history"
                            ),
                            "disclaimer": disease_result.get(
                                "disclaimer"
                            ),
                        }
                    )

                    continue

                # ------------------------------------------------
                # Save training provenance for the disease model.
                # ------------------------------------------------
                training_run = save_training_run(
                    db=db,
                    dataset_id=dataset.id,
                    training_period_start=date.fromisoformat(
                        disease_result[
                            "training_period_start"
                        ]
                    ),
                    training_period_end=date.fromisoformat(
                        disease_result[
                            "training_period_end"
                        ]
                    ),
                    model_name=disease_result[
                        "model_name"
                    ],
                    model_version=disease_result[
                        "model_version"
                    ],
                    feature_schema_version=disease_result[
                        "feature_schema_version"
                    ],
                    validation_metrics=disease_result[
                        "validation_metrics"
                    ],
                    test_metrics=disease_result[
                        "test_metrics"
                    ],
                )

                # ------------------------------------------------
                # Save disease forecast provenance.
                # ------------------------------------------------
                forecast = save_forecast(
                    db=db,
                    user_id=current_user.id,
                    dataset_id=dataset.id,
                    training_run_id=training_run.id,
                    forecast_type="disease",
                    horizon=disease_result[
                        "horizon"
                    ],
                    model_name=disease_result[
                        "model_name"
                    ],
                    model_version=disease_result[
                        "model_version"
                    ],
                    results={
                        "disease": disease_name,
                        "predictions": disease_result[
                            "predictions"
                        ],
                        "history": disease_result.get(
                            "history"
                        ),
                        "disclaimer": disease_result.get(
                            "disclaimer"
                        ),
                    },
                    performance_metrics={
                        "validation": disease_result[
                            "validation_metrics"
                        ],
                        "test": disease_result[
                            "test_metrics"
                        ],
                    },
                )

                disease_results.append(
                    {
                        "disease": disease_name,
                        "status": disease_result["status"],
                        "available": True,
                        "horizon": forecast.horizon,
                        "training_run_id": training_run.id,
                        "forecast_id": forecast.id,
                        "model_name": disease_result[
                            "model_name"
                        ],
                        "model_version": disease_result[
                            "model_version"
                        ],
                        "feature_schema_version": disease_result[
                            "feature_schema_version"
                        ],
                        "predictions": disease_result[
                            "predictions"
                        ],
                        "validation_metrics": disease_result[
                            "validation_metrics"
                        ],
                        "test_metrics": disease_result[
                            "test_metrics"
                        ],
                        "history": disease_result.get(
                            "history"
                        ),
                        "disclaimer": disease_result.get(
                            "disclaimer"
                        ),
                    }
                )

            # ----------------------------------------------------
            # Mark disease dataset as successfully processed.
            #
            # A dataset can be processed even when one or more
            # diseases have limited history. Those diseases remain
            # explicitly marked unavailable rather than receiving
            # fabricated forecast values.
            # ----------------------------------------------------
            dataset.status = "processed"

            db.commit()
            db.refresh(dataset)

        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Disease dataset processing failed due to a server or forecasting error.",
            ) from exc

        # ----------------------------------------------------
        # 15. Return disease processing response
        # ----------------------------------------------------
        return {
            "message": (
                "Disease dataset uploaded, validated, "
                "versioned, processed, and disease forecasts "
                "generated successfully."
            ),
            "dataset_id": dataset.id,
            "filename": safe_filename,
            "dataset_type": dataset_type,
            "row_count": validation_result["row_count"],
            "disease_count": disease_processing_result[
                "disease_count"
            ],
            "horizon": disease_processing_result[
                "horizon"
            ],
            "diseases": disease_results,
            "start_date": validation_result[
                "date_validation"
            ]["start_date"],
            "end_date": validation_result[
                "date_validation"
            ]["end_date"],
            "historical_days": validation_result[
                "historical_period"
            ]["days"],
            "disclaimer": (
                "Disease forecasts are hospital-specific and "
                "are based only on disease data uploaded by the "
                "hospital. Forecasts are intended for operational "
                "decision support and are not diagnosis or "
                "treatment recommendations."
            ),
            "user_id": current_user.id,
        }

    # --------------------------------------------------------
    # 16. Fallback response
    # --------------------------------------------------------
    return {
        "message": (
            "Dataset uploaded and validated successfully."
        ),
        "dataset_id": dataset.id,
        "filename": safe_filename,
        "dataset_type": dataset_type,
        "row_count": validation_result["row_count"],
        "start_date": validation_result[
            "date_validation"
        ]["start_date"],
        "end_date": validation_result[
            "date_validation"
        ]["end_date"],
        "historical_days": validation_result[
            "historical_period"
        ]["days"],
        "disclaimer": validation_result[
            "historical_period"
        ]["disclaimer"],
        "file_path": dataset.file_path,
        "user_id": current_user.id,
    }


@router.get("/")
def list_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user.id)
        .order_by(Dataset.upload_date.desc())
        .all()
    )

    return {
        "user_id": current_user.id,
        "count": len(datasets),
        "datasets": [
            {
                "id": dataset.id,
                "filename": dataset.filename,
                "dataset_type": dataset.dataset_type,
                "upload_date": dataset.upload_date,
                "start_date": dataset.start_date,
                "end_date": dataset.end_date,
                "status": dataset.status,
                "is_current": dataset.is_current,
                "version_number": dataset.version_number,
            }
            for dataset in datasets
        ],
    }


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "dataset_type": dataset.dataset_type,
        "upload_date": dataset.upload_date,
        "start_date": dataset.start_date,
        "end_date": dataset.end_date,
        "status": dataset.status,
        "is_current": dataset.is_current,
        "version_number": dataset.version_number,
        "user_id": dataset.user_id,
    }


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    if not dataset.file_path:
        raise HTTPException(
            status_code=404,
            detail="Dataset file is not available.",
        )

    # Download datasets stored in persistent Supabase Storage.
    if dataset.file_path.startswith("supabase://"):
        storage_path = dataset.file_path[len("supabase://"):]

        try:
            file_content = download_dataset_from_storage(storage_path)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail="Dataset file is not available.",
            ) from exc

        return Response(
            content=file_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{dataset.filename}"'
                )
            },
        )

    # Backward compatibility for existing local-file records.
    file_path = Path(dataset.file_path)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset file is not available.",
        )

    return FileResponse(
        path=file_path,
        filename=dataset.filename,
        media_type="text/csv",
    )


@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    file_path = (
        Path(dataset.file_path)
        if dataset.file_path
        else None
    )

    # Delete records associated with this dataset first.
    db.query(Forecast).filter(
        Forecast.dataset_id == dataset.id
    ).delete(
        synchronize_session=False
    )

    db.query(TrainingRun).filter(
        TrainingRun.dataset_id == dataset.id
    ).delete(
        synchronize_session=False
    )

    db.query(ResourceRecord).filter(
        ResourceRecord.dataset_id == dataset.id
    ).delete(
        synchronize_session=False
    )

    db.delete(dataset)
    db.commit()

    # Remove the stored CSV file after successful DB deletion.
    if dataset.file_path and dataset.file_path.startswith("supabase://"):
        storage_path = dataset.file_path[len("supabase://"):]

        try:
            delete_dataset_from_storage(storage_path)
        except Exception:
            # The database deletion has succeeded; storage cleanup can be retried.
            pass

    elif file_path and file_path.exists():
        file_path.unlink()

    return {
        "message": "Dataset deleted successfully.",
        "dataset_id": dataset_id,
    }