from assignment import (
    assign_heads_to_bikes, assign_people_to_bikes, bike_region,
    dedupe_detections, hip_from_keypoints, person_seat_point, smoothed_count,
)


def test_smoothed_count_needs_history_for_triple():
    assert smoothed_count([3]) == 2
    assert smoothed_count([3, 3]) == 2
    assert smoothed_count([3, 3, 3]) == 3


def test_smoothed_count_single_photo_can_be_triple():
    # regression: photos have ONE analysis and must still be able to flag 3 riders
    assert smoothed_count([3], min_history_for_triple=1) == 3


def test_smoothed_count_ignores_single_glitch_and_handles_empty():
    assert smoothed_count([2, 2, 3, 2, 2]) == 2
    assert smoothed_count([]) == 0


def test_hip_and_seat_point():
    kp = [[0, 0]] * 11 + [[10, 100], [30, 100]] + [[0, 0]] * 4
    conf = [0.0] * 11 + [0.9, 0.9] + [0.0] * 4
    assert hip_from_keypoints(kp, conf) == (20.0, 100.0)

    assert person_seat_point({"box": [0, 0, 100, 200], "hip": None}) == (50.0, 120.0)
    assert person_seat_point({"box": [0, 0, 100, 200], "hip": (7, 8)}) == (7, 8)


def test_hip_missing_keypoints_returns_none():
    assert hip_from_keypoints([[0, 0]] * 17, [0.0] * 17) is None


def test_bike_region_is_clamped_to_frame():
    x1, y1, x2, y2 = bike_region([10, 10, 110, 60], 120, 70)
    assert x1 >= 0 and y1 >= 0 and x2 <= 120 and y2 <= 70


def test_dedupe_detections_keeps_best():
    dets = [{"box": [0, 0, 10, 10], "conf": 0.4}, {"box": [0, 0, 10, 10], "conf": 0.8}]
    assert [d["conf"] for d in dedupe_detections(dets)] == [0.8]


def _rider(cx, hip_y=380):
    return {"box": [cx - 40, 150, cx + 40, 420], "hip": (cx, hip_y)}


def test_people_assigned_by_seat_point_to_nearest_bike():
    bikes = [{"box": [100, 300, 300, 450]}, {"box": [320, 300, 520, 450]}]
    people = [_rider(150), _rider(250), _rider(420)]

    counts, rider_boxes, others = assign_people_to_bikes(people, bikes)

    assert counts == {0: 2, 1: 1}
    assert others == []
    assert len(rider_boxes[0]) == 2


def test_pedestrian_is_not_a_rider():
    bikes = [{"box": [100, 300, 300, 450]}]
    walker = {"box": [700, 150, 780, 420], "hip": (740, 380)}

    counts, _, others = assign_people_to_bikes([walker], bikes)

    assert counts == {0: 0}
    assert others == [walker["box"]]


def test_hidden_pillion_head_counts_as_extra_rider():
    bikes = [{"box": [100, 300, 300, 450]}]
    rider = _rider(200)
    _, rider_boxes, others = assign_people_to_bikes([rider], bikes)

    own_head = {"box": [180, 140, 220, 180], "cls": 1, "conf": 0.9}
    hidden = {"box": [235, 260, 275, 300], "cls": 1, "conf": 0.9}   # above the bike, no body box

    owners = assign_heads_to_bikes([own_head, hidden], bikes, rider_boxes, others)

    extra = [o for o in owners if o[2]]
    assert len(owners) == 2 and len(extra) == 1


def test_pedestrian_head_is_ignored():
    bikes = [{"box": [100, 300, 300, 450]}]
    walker = {"box": [700, 150, 780, 420], "hip": (740, 380)}
    _, rider_boxes, others = assign_people_to_bikes([walker], bikes)

    head = {"box": [720, 150, 760, 190], "cls": 1, "conf": 0.9}

    assert assign_heads_to_bikes([head], bikes, rider_boxes, others) == []


# ------------------------------------------------ mannequins / bystanders

SCOOTER = {"box": [170, 400, 317, 510]}          # parked scooter, 110 px tall


def _standing(cx, head_y=300, hip_y=395):
    """A person standing BEHIND the scooter (e.g. a shop mannequin)."""
    return {"box": [cx - 40, head_y - 18, cx + 40, head_y + 125],
            "head": (cx, head_y), "hip": (cx, hip_y)}


def test_mannequins_behind_a_parked_scooter_are_not_riders():
    """Regression: 4 shop mannequins were reported as 'RIDERS: 4 | TRIPLE RIDING'."""
    from assignment import assign_people_to_bikes

    mannequins = [_standing(x) for x in (180, 215, 250, 285)]

    counts, rider_boxes, others = assign_people_to_bikes(mannequins, [SCOOTER])

    assert counts == {0: 0}
    assert rider_boxes == {0: []} and len(others) == 4


def test_real_rider_on_the_same_scooter_is_kept():
    from assignment import assign_people_to_bikes

    rider = {"box": [190, 300, 260, 470], "head": (225, 330), "hip": (225, 430)}

    counts, _, others = assign_people_to_bikes([rider], [SCOOTER])

    assert counts == {0: 1} and others == []


def test_head_far_above_bike_needs_BOTH_conditions_to_reject():
    """A tall rider on a short bike box (close-up photo) must still count."""
    from assignment import is_plausible_rider

    bike = {"box": [0, 500, 300, 600]}                       # 100 px tall
    tall_rider = {"box": [50, 350, 250, 620], "head": (150, 380)}   # 120 above = 1.2 bikes, 0.44 person
    mannequin = {"box": [50, 380, 130, 520], "head": (90, 395)}     # 105 above = 1.05 bikes, 0.75 person

    assert is_plausible_rider(tall_rider, bike) is True
    assert is_plausible_rider(mannequin, bike) is False


def test_limits_measured_on_real_photos_still_pass():
    """Riders measured at <= 0.59 bike-heights / 0.51 person-heights above the bike."""
    from assignment import is_plausible_rider

    bike = {"box": [0, 1000, 500, 1200]}                     # 200 px tall
    rider = {"box": [100, 800, 300, 1150], "head": (200, 882)}      # 118 above = 0.59 / 0.34

    assert is_plausible_rider(rider, bike) is True


def test_heads_of_bystanders_are_ignored_after_rejection():
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    mannequins = [_standing(215)]
    _, rider_boxes, others = assign_people_to_bikes(mannequins, [SCOOTER])

    head = {"box": [195, 282, 235, 318], "cls": 0, "conf": 0.9}

    assert assign_heads_to_bikes([head], [SCOOTER], rider_boxes, others) == []


def test_claims_report_which_rider_owns_which_head():
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    bike = {"box": [100, 300, 300, 450]}
    riders = [
        {"box": [110, 150, 190, 420], "head": (150, 170), "hip": (150, 380)},
        {"box": [200, 150, 280, 420], "head": (240, 170), "hip": (240, 380)},
    ]
    _, rider_boxes, others = assign_people_to_bikes(riders, [bike])
    head = {"box": [130, 150, 170, 190], "cls": 1, "conf": 0.9}      # only rider 1 has a head box

    claims = []
    assign_heads_to_bikes([head], [bike], rider_boxes, others, claims)

    assert len(claims) == 1 and claims[0][0] == riders[0]["box"]

def test_fullframe_heads_fill_gaps_and_respect_regions():
    from assignment import merge_fullframe_heads

    crop = [{"box": [100, 100, 200, 200], "conf": 0.7, "cls": 0}]
    full = [
        {"box": [102, 98, 198, 202], "conf": 0.8, "cls": 0},      # same head, same class
        {"box": [300, 100, 400, 200], "conf": 0.8, "cls": 0},     # missed by the crop
        {"box": [900, 100, 990, 190], "conf": 0.9, "cls": 0},     # outside every bike region
    ]
    regions = [(0, 0, 500, 400)]

    merged = merge_fullframe_heads(crop, full, regions)

    assert len(merged) == 2
    assert merged[0]["box"] == [100, 100, 200, 200]               # the crop's box is kept
    assert merged[1]["box"] == [300, 100, 400, 200]


def test_fullframe_head_overrides_a_conflicting_crop_call_only_when_clearly_better():
    from assignment import merge_fullframe_heads

    regions = [(0, 0, 500, 400)]
    crop = [{"box": [100, 100, 200, 200], "conf": 0.45, "cls": 1}]    # crop says Without

    better = [{"box": [100, 100, 200, 200], "conf": 0.80, "cls": 0}]  # whole frame: With
    close = [{"box": [100, 100, 200, 200], "conf": 0.47, "cls": 0}]

    assert merge_fullframe_heads(crop, better, regions)[0]["cls"] == 0
    assert merge_fullframe_heads(crop, close, regions)[0]["cls"] == 1


def test_confident_whole_frame_helmet_beats_a_unsure_crop_without_helmet():
    from assignment import merge_fullframe_heads

    regions = [(0, 0, 500, 400)]
    crop = [{"box": [100, 100, 200, 200], "conf": 0.60, "cls": 1}]     # crop: Without, unsure
    full = [{"box": [100, 100, 200, 200], "conf": 0.55, "cls": 0}]     # whole frame: With

    assert merge_fullframe_heads(crop, full, regions)[0]["cls"] == 0

    sure = [{"box": [100, 100, 200, 200], "conf": 0.90, "cls": 1}]     # crop: Without, very sure
    assert merge_fullframe_heads(sure, full, regions)[0]["cls"] == 1   # needs a clear win to flip