"""Safeguards against counting too many riders (and that real riders survive them)."""

from assignment import (
    assign_heads_to_bikes, assign_people_to_bikes,
    drop_headless_people, merge_nested_heads,
)

BIKE = {"box": [200, 250, 400, 420], "conf": 0.9, "track_id": 1}


def person(box, head, hip, conf):
    return {"box": box, "head": head, "hip": hip, "conf": conf}


def head(box, conf=0.8, cls=1):
    return {"box": box, "conf": conf, "cls": cls}


def test_face_box_inside_helmet_box_is_one_head():
    helmet = head([100, 100, 160, 160], 0.8)
    face = head([110, 110, 150, 150], 0.5)
    assert merge_nested_heads([face, helmet]) == [helmet]


def test_two_separate_heads_are_both_kept():
    a, b = head([100, 100, 140, 140]), head([150, 100, 190, 140])
    assert len(merge_nested_heads([a, b])) == 2


def test_body_piece_without_head_is_dropped():
    ghost = person([300, 200, 360, 380], None, None, 0.35)
    real = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    assert drop_headless_people([real, ghost], []) == [real]


def test_headless_but_confident_person_is_kept():
    sure = person([300, 200, 360, 380], None, None, 0.80)
    assert drop_headless_people([sure], []) == [sure]


def test_headless_person_kept_when_helmet_model_sees_a_head():
    p = person([300, 200, 360, 380], None, None, 0.35)
    h = head([310, 190, 350, 235])
    assert drop_headless_people([p], [h]) == [p]


def test_far_away_head_is_not_a_hidden_rider():
    rider = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    _, rider_boxes, other = assign_people_to_bikes([rider], [BIKE])
    far = head([-20, 120, 20, 160])            # in the bike's wide zone but far from the rider
    owners = assign_heads_to_bikes([far], [BIKE], rider_boxes, other)
    assert not any(extra for _, _, extra in owners)


def test_pillion_head_next_to_rider_still_counts():
    rider = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    _, rider_boxes, other = assign_people_to_bikes([rider], [BIKE])
    own = head([232, 100, 268, 140])
    pillion = head([290, 105, 330, 145])        # right beside the rider, no body box
    owners = assign_heads_to_bikes([own, pillion], [BIKE], rider_boxes, other)
    assert sum(1 for _, _, extra in owners if extra) == 1


# ---------- round 4: bystanders, pedestrians, duplicate bike boxes, helmet re-checks ----------

from assignment import dedupe_nested_bikes  # noqa: E402
from head_check import verify_with_decision  # noqa: E402


def test_person_sitting_behind_a_bike_is_not_its_rider():
    # hips fall inside the (widened) bike box, but the body is entirely above the bike
    bystander = person([220, 60, 290, 240], (255, 80), (255, 245), 0.9)
    counts, _, others = assign_people_to_bikes([bystander], [BIKE])
    assert counts[0] == 0 and len(others) == 1


def test_tiny_pedestrian_is_not_counted_next_to_real_riders():
    rider_a = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    rider_b = person([300, 105, 370, 405], (335, 125), (335, 365), 0.85)
    walker = person([330, 300, 345, 340], (337, 305), (337, 330), 0.7)   # far away, tiny
    counts, _, others = assign_people_to_bikes([rider_a, rider_b, walker], [BIKE])
    assert counts[0] == 2 and len(others) == 1


def test_child_pillion_is_still_counted():
    adult = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    child = person([290, 220, 330, 380], (310, 235), (310, 350), 0.6)    # ~30% of the adult's area
    counts, _, _ = assign_people_to_bikes([adult, child], [BIKE])
    assert counts[0] == 2


def test_tiny_head_far_down_the_road_is_not_a_hidden_rider():
    rider = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    _, rider_boxes, other = assign_people_to_bikes([rider], [BIKE])
    own = head([232, 100, 268, 140])                 # 36 px wide
    walker_head = head([300, 110, 310, 122])         # 10 px wide, close to the rider
    owners = assign_heads_to_bikes([own, walker_head], [BIKE], rider_boxes, other)
    assert not any(extra for _, _, extra in owners)


def test_partial_box_over_a_bike_is_dropped():
    full = {"box": [200, 250, 700, 520], "conf": 0.9, "track_id": 1}
    part = {"box": [200, 260, 400, 520], "conf": 0.5, "track_id": 2}
    assert dedupe_nested_bikes([part, full]) == [full]


def test_two_neighbouring_bikes_are_both_kept():
    a = {"box": [0, 250, 300, 520], "conf": 0.9, "track_id": 1}
    b = {"box": [250, 250, 600, 520], "conf": 0.8, "track_id": 2}
    assert len(dedupe_nested_bikes([a, b])) == 2


def test_dark_helmet_called_without_is_flipped_by_zoomed_look():
    zoomed = [{"box": [0, 0, 1, 1], "conf": 0.6, "cls": 0},
              {"box": [0, 0, 1, 1], "conf": 0.3, "cls": 1}]
    assert verify_with_decision(zoomed, 0, 1, 0.35)["cls"] == 0


def test_bare_head_stays_without_when_zoomed_look_agrees():
    zoomed = [{"box": [0, 0, 1, 1], "conf": 0.7, "cls": 1}]
    assert verify_with_decision(zoomed, 0, 1, 0.35) is None


def test_rider_claims_the_head_nearest_their_own_keypoint():
    # woman (left) and a small child head both lie in the woman's head zone; the
    # child's head is nearer the top-centre of her box, but her OWN head keypoint
    # is at the left head.
    woman = person([139, 33, 288, 514], (183, 85), (185, 296), 0.76)
    bike = {"box": [101, 200, 495, 577], "conf": 0.9}
    woman_head = head([136, 23, 205, 139], 0.71, 0)
    child_head = head([198, 62, 242, 143], 0.33, 0)
    _, rider_boxes, other = assign_people_to_bikes([woman], [bike])

    plain = assign_heads_to_bikes([woman_head, child_head], [bike], rider_boxes, other, [])
    keyed = assign_heads_to_bikes([woman_head, child_head], [bike], rider_boxes, other, [],
                                  people=[woman])

    own_plain = [d for d, _, extra in plain if not extra]
    own_keyed = [d for d, _, extra in keyed if not extra]
    assert own_plain == [child_head]       # old behaviour picked the child's head
    assert own_keyed == [woman_head]


def test_rider_without_a_found_head_gets_an_estimated_head_box():
    from assignment import unresolved_rider_heads

    riders = [[100, 100, 200, 400], [220, 120, 320, 400]]
    people = [{"box": riders[0], "head": (150, 130)}, {"box": riders[1], "head": None}]
    found_heads = [[130, 105, 170, 145]]            # only the first rider's head

    missing = unresolved_rider_heads(riders, people, found_heads)

    assert len(missing) == 1                         # the second rider only
    cx = (missing[0][0] + missing[0][2]) / 2
    assert 220 <= cx <= 320 and missing[0][1] < 200  # near the top of that rider
    assert unresolved_rider_heads(riders, people, []) != []