def volume_collate(items):
    """Keep complete, variable-size grids as a list for independent encoding."""
    if not items:
        raise ValueError("Empty batch")
    return items
