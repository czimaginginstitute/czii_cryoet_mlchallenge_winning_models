ANGSTROMS_IN_PIXEL = 10.012

TARGET_CLASSES = (
    {
        "name": "apo-ferritin",
        "is_particle": True,
        "pdb_id": "4V1W",
        "label": 1,
        "color": [  0, 117, 220, 255],
        "radius": 60,
        "map_threshold": 0.0418,
        "score_weight": 1
    },
    {
        "name": "beta-amylase",
        "is_particle": True,
        "pdb_id": "1FA2",
        "label": 2,
        "color": [153,  63,   0, 255],
        "radius": 65,
        "map_threshold": 0.035,
        "score_weight": 0
    },
    {
        "name": "beta-galactosidase",
        "is_particle": True,
        "pdb_id": "6X1Q",
        "label": 3,
        "color": [ 76,   0,  92, 255],
        "radius": 90,
        "map_threshold": 0.0578,
        "score_weight": 2
    },
    {
        "name": "ribosome",
        "is_particle": True,
        "pdb_id": "6EK0",
        "label": 4,
        "color": [  0,  92,  49, 255],
        "radius": 150,
        "map_threshold": 0.0374,
        "score_weight": 1
    },
    {
        "name": "thyroglobulin",
        "is_particle": True,
        "pdb_id": "6SCJ",
        "label": 5,
        "color": [ 43, 206,  72, 255],
        "radius": 130,
        "map_threshold": 0.0278,
        "score_weight": 2
    },
    {
        "name": "virus-like-particle",
        "is_particle": True,
        "label": 6,
        "color": [255, 204, 153, 255],
        "radius": 135,
        "map_threshold": 0.201,
        "score_weight": 1
    }
)

CLASS_INDEX_TO_CLASS_NAME = {i: c["name"] for i, c in enumerate(TARGET_CLASSES)}
TARGET_SIGMAS = [c["radius"] / ANGSTROMS_IN_PIXEL for c in TARGET_CLASSES]
WEIGHTS = {c['name']: c['score_weight'] for c in TARGET_CLASSES}