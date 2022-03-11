rooms = [
    (
        "No key no obstacle close to the door",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "top-right",
            "obstacle_locations": [],
            "need_key": False,
        },
    ),
    (
        "No key no obstacle far from the door",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [],
            "need_key": False,
        },
    ),
    (
        "No key no obstacle",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, 4 // 2), "top-left", "bottom-left"],
            "need_key": False,
        },
    ),
    (
        "No key but obstacles",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, 4 // 2), "top-left", "bottom-right"],
            "need_key": False,
        },
    ),
    (
        "Key bottom-right but no obstacles",
        {
            "door_location": "top-middle",
            "key_location": "bottom-right",
            "agent_location": "bottom-middle",
            "obstacle_locations": [],
            "need_key": True,
        },
    ),
    (
        "Key bottom right + obstacles",
        {
            "door_location": "top-left",
            "key_location": "bottom-right",
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, 4 // 2), "top-right", "bottom-left"],
            "need_key": True,
        },
    ),
]
