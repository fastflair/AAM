"""
film_maker — idea → animated film, rendered by MiniMax H3 (via Wan2GP).

Notebook surface (keep the notebook tiny):

    from film_maker import FILM_CONFIG, plan_film, produce_film, regenerate_scenes

    FILM_CONFIG["registers"] = ["scifi", "drama"]
    FILM_CONFIG["target_minutes"] = 16

    plan_path = plan_film("A lighthouse keeper discovers the light is a door.",
                          FILM_CONFIG)
    # ...open plan.json, edit anything...
    produce_film(plan_path, FILM_CONFIG)

    # Re-roll specific scenes after watching:
    regenerate_scenes(plan_path, FILM_CONFIG, scene_ids=[4, 11])
"""
from .config import FILM_CONFIG
from .pipeline import (plan_film, produce_film, find_plan,
                       regenerate_scenes,
                       regenerate_shots, list_projects,
                       prune_unused_stills)

__all__ = ["FILM_CONFIG", "plan_film", "produce_film", "find_plan",
           "regenerate_scenes", "regenerate_shots", "list_projects",
           "prune_unused_stills"]
