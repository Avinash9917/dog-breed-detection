import json
from typing import Dict, Any, Optional
from app.config import BREEDS_FILE

class BreedTraitService:
    def __init__(self, breeds_path=BREEDS_FILE):
        self.breeds_path = breeds_path
        self.traits: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.load_traits()

    def load_traits(self) -> None:
        if self.breeds_path.exists():
            with open(self.breeds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.meta = data.get("_meta", {})
                self.traits = {k: v for k, v in data.items() if k != "_meta"}
        else:
            self.traits = {}
            self.meta = {}

    def get_traits(self, breed_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve traits for a given breed name.
        Uses exact string match, with fallback normalized case.
        """
        if breed_name in self.traits:
            return self.traits[breed_name]
        
        # Fallback case-insensitive match
        lower_map = {k.lower(): v for k, v in self.traits.items()}
        return lower_map.get(breed_name.lower(), None)


trait_service = BreedTraitService()
