CLIP_ANALYSIS_PROMPTS: dict[str, str] = {
    "scene_analysis": (
        "Observe these video frames and describe the basic contents of the scene.\n\n"
        "Describe:\n"
        "1. The environment/setting\n"
        "2. Any people present\n"
        "3. Key objects or items\n"
        "4. Notable activities happening\n"
        "5. Important visual elements"
    ),
    "temporal_analysis": (
        "Review how this scene changes across these sequential frames.\n\n"
        "Describe:\n"
        "1. Initial scene state\n"
        "2. Key changes or movements\n"
        "3. Order of events/actions\n"
        "4. Final scene state\n"
        "5. Any patterns in movement/behavior"
    )
}
