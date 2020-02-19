
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 5

# ANCHORS_GROUP = {
#     13: [[169.5, 330], [252, 206], [219, 242]],
#     26: [[104, 167], [274, 171], [93.5, 88]],
#     52: [[50, 97.5], [271, 143], [42, 22]]
# }

ANCHORS_GROUP = {
    13: [[168, 247], [177, 176], [322.5, 198]],
    26: [[109, 159.5], [211, 130], [76, 71]],
    52: [[61, 50], [176, 294], [62, 139]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
