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


# ------------------------------------------------ far-away background people

def _bike_with_two_riders():
    bike = {"box": [100, 300, 400, 580]}                       # 280 px tall
    riders = [
        {"box": [120, 200, 220, 520], "head": (170, 240), "hip": (170, 430)},
        {"box": [230, 200, 330, 520], "head": (280, 240), "hip": (280, 430)},
    ]
    return bike, riders


def test_small_background_heads_are_not_hidden_pillions():
    """Regression: 2 tiny pedestrians far behind the bike made 'RIDERS: 5'."""
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    bike, riders = _bike_with_two_riders()
    _, rider_boxes, others = assign_people_to_bikes(riders, [bike])
    real = [{"box": [150, 215, 190, 265], "cls": 1, "conf": 0.9},        # 50 px heads
            {"box": [260, 215, 300, 265], "cls": 1, "conf": 0.9}]
    far = [{"box": [200, 250, 212, 262], "cls": 1, "conf": 0.8},         # 12 px: tiny
           {"box": [340, 250, 352, 262], "cls": 1, "conf": 0.8}]

    owners = assign_heads_to_bikes(real + far, [bike], rider_boxes, others)

    assert sum(1 for _, _, extra in owners if extra) == 0


def test_true_hidden_pillion_of_normal_size_is_still_counted():
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    bike, riders = _bike_with_two_riders()
    _, rider_boxes, others = assign_people_to_bikes(riders, [bike])
    heads = [{"box": [150, 215, 190, 265], "cls": 1, "conf": 0.9},
             {"box": [260, 215, 300, 265], "cls": 1, "conf": 0.9},
             {"box": [205, 225, 245, 275], "cls": 0, "conf": 0.9}]       # hidden, same size

    owners = assign_heads_to_bikes(heads, [bike], rider_boxes, others)

    assert sum(1 for _, _, extra in owners if extra) == 1


def test_tiny_person_next_to_a_bike_is_not_a_rider():
    from assignment import is_plausible_rider

    bike = {"box": [0, 300, 300, 580]}                          # 280 px
    pedestrian = {"box": [100, 250, 120, 320], "head": (110, 258)}   # 70 px = 0.25 bikes

    assert is_plausible_rider(pedestrian, bike) is False


def test_people_sitting_behind_a_parked_bike_are_not_its_riders():
    """Regression: two men on steps behind a parked bike were counted as '2 riders'."""
    from assignment import assign_people_to_bikes

    bike = {"box": [485, 280, 920, 675]}
    sitters = [{"box": [550, 130, 640, 262], "head": (590, 150), "hip": (595, 235)},
               {"box": [640, 135, 720, 266], "head": (670, 155), "hip": (680, 240)}]

    counts, rider_boxes, others = assign_people_to_bikes(sitters, [bike])

    assert counts == {0: 0} and len(others) == 2


def test_rider_with_legs_over_the_bike_still_counts():
    from assignment import assign_people_to_bikes

    bike = {"box": [100, 300, 400, 560]}
    rider = {"box": [150, 160, 260, 540], "head": (200, 190), "hip": (205, 420)}

    assert assign_people_to_bikes([rider], [bike])[0] == {0: 1}


# ------------------------------------------------ crowded riders (real numbers from triple_3.jpg)

TRIPLE_3_BIKE = {"box": [1276, 2255, 4271, 3615]}
TRIPLE_3_PEOPLE = [        # order as the pose model returned them: rear rider FIRST
    {"box": [2727, 1346, 4240, 3622], "head": (3572, 1511), "hip": (3647, 2948)},    # rear
    {"box": [1544, 1288, 2924, 3586], "head": (2436, 1549), "hip": (2648, 2904)},    # front
    {"box": [2222, 1278, 3454, 3601], "head": (3035, 1528), "hip": (3063, 2959)},    # middle
]
TRIPLE_3_HEADS = [
    {"box": [2843, 1189, 3391, 1795], "cls": 0, "conf": 0.37},     # middle rider's helmet
    {"box": [3423, 1001, 3980, 1528], "cls": 0, "conf": 0.52},     # rear rider's helmet
]


def test_crowded_riders_keep_their_own_helmets_and_are_counted_once():
    """Regression: the rear rider took the middle rider's helmet -> '4 riders' (3 real)."""
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    counts, rider_boxes, others = assign_people_to_bikes(TRIPLE_3_PEOPLE, [TRIPLE_3_BIKE])
    claims = []
    owners = assign_heads_to_bikes(TRIPLE_3_HEADS, [TRIPLE_3_BIKE], rider_boxes, others,
                                   claims, TRIPLE_3_PEOPLE)

    assert counts == {0: 3}
    assert sum(1 for _, _, extra in owners if extra) == 0          # no phantom 4th rider
    owner_of = {tuple(body): det["box"] for body, det in claims}
    assert owner_of[(2727, 1346, 4240, 3622)] == [3423, 1001, 3980, 1528]    # rear -> rear helmet
    assert owner_of[(2222, 1278, 3454, 3601)] == [2843, 1189, 3391, 1795]    # middle -> middle helmet
    assert (1544, 1288, 2924, 3586) not in owner_of                           # front: no head found


def test_without_pose_head_points_matching_still_works():
    from assignment import assign_heads_to_bikes, assign_people_to_bikes

    bike = {"box": [100, 300, 300, 450]}
    riders = [{"box": [110, 150, 190, 420], "head": (150, 170), "hip": (150, 380)},
              {"box": [200, 150, 280, 420], "head": (240, 170), "hip": (240, 380)}]
    _, rider_boxes, others = assign_people_to_bikes(riders, [bike])
    heads = [{"box": [220, 150, 260, 190], "cls": 0, "conf": 0.9},
             {"box": [130, 150, 170, 190], "cls": 1, "conf": 0.9}]

    claims = []
    assign_heads_to_bikes(heads, [bike], rider_boxes, others, claims)       # no people passed

    assert len(claims) == 2 and {c[1]["cls"] for c in claims} == {0, 1}