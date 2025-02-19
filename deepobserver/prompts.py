CLIP_ANALYSIS_PROMPTS = {
    "scene_analysis": """
    You are a computer vision AI analyzing a security camera feed. Provide an exhaustive analysis of the scene.
    
    Analyze and describe in detail:
    1. Environment:
       - Room/space type and layout
       - Lighting conditions
       - Entry/exit points
       - Furniture arrangement
       
    2. People (CRITICAL - describe in detail):
       - Number of people
       - Physical descriptions
       - Clothing details
       - Positions in the space
       - Body language and behavior
       - Direction of movement
       - Interactions with others/objects
       
    3. Objects:
       - All visible objects
       - Their positions and arrangements
       - State (in use, idle, moved)
       - Relationship to people/other objects
       
    4. Activities:
       - What each person is doing
       - Group dynamics
       - Use of objects/space
       - Movement patterns
       
    5. Notable Elements:
       - Anything unusual or significant
       - Security-relevant details
       - Changes from previous observations
    
    Provide specific details and locations. Do not omit any information about people or activities.
    """,

    "temporal_analysis": """
    Analyze the temporal changes in this sequence of frames. Focus on:
    1. Movement patterns of people
    2. Changes in object positions
    3. Entry and exit of people from the scene
    4. Changes in behavior or activity
    5. Any patterns or repeated actions
    
    Remember: Detailed descriptions of people and their actions are required for security purposes.
    """,

    "semantic_analysis": """
    Based on the scene analysis: {scene_analysis}
    And temporal analysis: {temporal_analysis}
    
    Provide a comprehensive security-focused interpretation:
    1. Assess all human activity and behavior
    2. Evaluate potential security concerns
    3. Identify patterns or anomalies
    4. Analyze interactions and relationships
    5. Flag any suspicious or unusual activity
    """
} 