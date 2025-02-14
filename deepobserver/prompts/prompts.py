CLIP_ANALYSIS_PROMPTS: dict[str, str] = {
    "scene_analysis": """
        Analyze these security camera frames and describe the basic contents of the scene.

        Describe:
        1. The surveillance environment/setting (indoor/outdoor location, entry/exit points, camera angle)
        2. Any people present (number, appearance, positioning, behavior)
        3. Key objects or items (vehicles, bags, tools, equipment)
        4. Observable activities (interactions, movements, tasks being performed)
        5. Security-relevant visual elements (access points, restricted areas, security equipment)
    """,
    "temporal_analysis": """
        Review how this security footage changes across these sequential frames.

        Describe:
        1. Initial state of the monitored area
        2. Key changes in movement or positioning (people, objects, vehicles)
        3. Chronological sequence of events/actions
        4. Final state of the scene
        5. Any patterns or anomalies in movement/behavior that warrant attention
    """,
    "semantic_analysis": """
        Based on the SCENE ANALYSIS: {scene_analysis}
        And the TEMPORAL ANALYSIS: {temporal_analysis}

        Provide a meaningful interpretation:
        1. Purpose and intent of observed activities
        2. Relationships and interactions between people/objects
        3. Behavioral patterns and their significance
        4. Context-specific implications of the activities
        5. Any security or safety concerns warranting attention
    """
}