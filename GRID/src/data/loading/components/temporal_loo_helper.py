"""
Helper utilities for temporal leave-one-out (LOO) evaluation.

For temporal LOO evaluation:
- Training sequences are shortened (last 2 items held out)
- Validation: second-to-last held-out item per user
- Testing: last held-out item per user

This module provides utilities to load held-out items from JSON metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalLOOManager:
    """Manager for temporal LOO evaluation data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the temporal LOO manager.
        
        Args:
            data_dir: Path to the data directory containing split_config.json,
                     validation_items.json, and test_items.json
        """
        self.data_dir = Path(data_dir)
        self._load_split_config()
        self._load_validation_items()
        self._load_test_items()
    
    def _load_split_config(self):
        """Load split configuration to verify temporal LOO is being used."""
        config_path = self.data_dir / "split_config.json"
        if not config_path.exists():
            logger.warning(f"Split config not found at {config_path}")
            self.split_strategy = None
            return
        
        with open(config_path, 'r') as f:
            self.split_config = json.load(f)
            self.split_strategy = self.split_config.get("strategy")
            logger.info(f"Loaded split strategy: {self.split_strategy}")
    
    def _load_validation_items(self):
        """Load validation held-out items mapping."""
        val_path = self.data_dir / "validation_items.json"
        if not val_path.exists():
            logger.warning(f"Validation items not found at {val_path}")
            self.validation_items = {}
            return
        
        with open(val_path, 'r') as f:
            self.validation_items = json.load(f)
        logger.info(f"Loaded {len(self.validation_items)} validation items")
    
    def _load_test_items(self):
        """Load test held-out items mapping."""
        test_path = self.data_dir / "test_items.json"
        if not test_path.exists():
            logger.warning(f"Test items not found at {test_path}")
            self.test_items = {}
            return
        
        with open(test_path, 'r') as f:
            self.test_items = json.load(f)
        logger.info(f"Loaded {len(self.test_items)} test items")
    
    def is_temporal_loo(self) -> bool:
        """Check if this data uses temporal LOO evaluation."""
        return self.split_strategy == "temporal_loo"
    
    def get_validation_item(self, user_id: str) -> Optional[Dict]:
        """Get validation held-out item for a user."""
        return self.validation_items.get(str(user_id))
    
    def get_test_item(self, user_id: str) -> Optional[Dict]:
        """Get test held-out item for a user."""
        return self.test_items.get(str(user_id))
    
    def has_validation_for_user(self, user_id: str) -> bool:
        """Check if user has a validation item."""
        item = self.validation_items.get(str(user_id))
        return item is not None
    
    def has_test_for_user(self, user_id: str) -> bool:
        """Check if user has a test item."""
        item = self.test_items.get(str(user_id))
        return item is not None
