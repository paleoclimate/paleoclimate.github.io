"""Present-day coordinates resolve to the Zahirovic 2022 PlateID."""

from plate_lookup import load_plates, plate_id_at


def test_known_coordinates_resolve_to_their_plate():
    plates = load_plates()
    expected = {
        (-40.25, -10.5): 201,
        (10.0, 5.5): 701,
        (1.57, 33.06): 714,
        (-65.732, -20.0): 201,
        (-72.6, 8.65): 2030,
        (28.9484, 28.0): 715,
        (-72.099, -49.044): 28800,
        (0.0, 0.0): 701,
        (151.0, -33.0): 801,
        (179.5, 0.0): 901,
        (-179.5, 0.0): 901,
    }
    for (longitude, latitude), plate_id in expected.items():
        assert plate_id_at(longitude, latitude, plates) == plate_id


def test_a_coordinate_outside_the_geographic_range_is_not_a_plate():
    plates = load_plates()
    assert plate_id_at(0, 91, plates) is None
    assert plate_id_at(181, 0, plates) is None
