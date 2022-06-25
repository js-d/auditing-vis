methods = [
    "intgrad",
    "gradcam",
    "gbp",
    "guided_gradcam",
    "car.layer2_2_conv1",
    "car.layer3_0_conv3",
    "car.layer3_2_conv2",
    "car.layer3_3",
    "car.layer3_4_conv3",
    "car.layer4_0_conv3",
    "car.layer4_2_conv3",
]

anomaly_types = [
    "adversarial_linf",
    "adversarial_l2",
    "trained_on_SIN",
    "spurious",
    "randomized",
    "backdoor",
    "blur",
    "missing",
]

lpips_nets = ["alex", "vgg", "squeeze"]

core_methods = [
    "car.layer2_2_conv1",
    "car.layer3_2_conv2",
    "car.layer3_3",
    "car.layer4_0_conv3",
]

all_models = [
    "randomized_smoothing_noise_0.50",
    "adversarial_l2_0.01",
    "adversarial_l2_1",
    "adversarial_linf_4",
    "backdoor_0",
    "backdoor_2",
    "backdoor_4",
    "blur",
    "missing_400",
    "null_1",
    "null_11",
    "null_2",
    "null_4",
    "null_6",
    "null_8",
    "spurious_0.1",
    "spurious_0.5",
    "trained_on_SIN",
    "adversarial_l2_0.1",
    "adversarial_l2_5",
    "adversarial_linf_8",
    "backdoor_1",
    "backdoor_3",
    "backdoor_5",
    "missing_218",
    "missing_414",
    "null_10",
    "null_12",
    "null_3",
    "null_5",
    "null_7",
    "null_9",
    "randomized_smoothing_noise_0.25",
    "randomized_smoothing_noise_1.00",
    "spurious_0.3",
    "spurious_1.0",
]

null_models = [m for m in all_models if m.startswith("null")]

anomalous_models = [m for m in all_models if not m.startswith("null")]


localization_models = [
    m
    for m in anomalous_models
    if m.startswith("spurious")
    or m.startswith("backdoor")
    or m.startswith("blur")
    or m.startswith("missing")
]

core_localization_models = [
    "spurious_0.5",
    "spurious_0.1",
    "blur",
    "missing_414",
    "backdoor_5",
    "backdoor_1",
]
