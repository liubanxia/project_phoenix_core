from core.result_fusion import fuse_results


def test_result_fusion_normalises_geometry_scores_and_metadata():
    raw = {
        "fracture_detector": {
            "lesions": [
                {
                    "label_name": "fracture",
                    "score": "0.91",
                    "series_uid": "1.2.3",
                    "image_index": "7",
                    "geometry": {"box": [10, 20, 30, 40]},
                    "point": [20, 30],
                    "custom_flag": "synthetic",
                }
            ]
        },
        "ct_segmenter": {
            "lesions": [
                {
                    "finding": "nodule",
                    "confidence": 0.82,
                    "box_3d": [1, 2, 3, 4, 5, 6],
                    "world_point_lps": [12.5, -3.0, 44.0],
                    "voxel_count": 128,
                    "geometry_mode": "3d",
                }
            ]
        },
    }

    result = fuse_results(raw)

    assert result.raw_model_results is raw
    assert result.warnings == []
    assert len(result.lesions) == 2

    fracture = result.lesions[0]
    assert fracture.label == "fracture"
    assert fracture.confidence == 0.91
    assert fracture.image_index == 7
    assert fracture.box == (10.0, 20.0, 30.0, 40.0)
    assert fracture.point == (20.0, 30.0)
    assert fracture.source_model == "fracture_detector"
    assert fracture.metadata["custom_flag"] == "synthetic"

    nodule = result.lesions[1]
    assert nodule.label == "nodule"
    assert nodule.box_3d == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert nodule.world_point_lps == (12.5, -3.0, 44.0)
    assert nodule.voxel_count == 128
    assert nodule.geometry_mode == "3d"
    assert nodule.source_model == "ct_segmenter"


def test_result_fusion_ignores_non_mapping_and_non_list_payloads():
    result = fuse_results(
        {
            "bad_model": "not-a-dict",
            "bad_lesions": {"lesions": "not-a-list"},
            "mixed": {"lesions": [None, "bad", {"label": 3, "confidence": "bad"}]},
        }
    )

    assert len(result.lesions) == 1
    assert result.lesions[0].label == "异常候选灶"
    assert result.lesions[0].confidence == 0.0
    assert result.lesions[0].source_model == "mixed"


def test_result_fusion_tolerates_invalid_geometry_and_uses_label_score():
    result = fuse_results(
        {
            "synthetic_detector": {
                "lesions": [
                    {
                        "name": "synthetic_candidate",
                        "label_score": "0.73",
                        "image_index": "not-an-int",
                        "point": ["bad", 12],
                        "box": [1, "bad", 3, 4],
                        "box_3d": "not-a-box",
                        "world_point_lps": [1, 2],
                    }
                ]
            }
        }
    )

    assert result.warnings == []
    assert len(result.lesions) == 1
    lesion = result.lesions[0]
    assert lesion.label == "synthetic_candidate"
    assert lesion.confidence == 0.73
    assert lesion.image_index is None
    assert lesion.point is None
    assert lesion.box is None
    assert lesion.box_3d is None
    assert lesion.world_point_lps is None
    assert lesion.source_model == "synthetic_detector"
