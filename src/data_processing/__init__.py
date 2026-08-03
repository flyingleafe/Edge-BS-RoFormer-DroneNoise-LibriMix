"""Data processing utilities for various audio datasets.

Most functions have moved to the subpackages:

- ``sources.*`` — the uniform external-dataset registry
- ``mixing.*`` — the pure mixture-render cores
- ``derivations.*`` — derived-dataset pipeline specs
- ``online_mixing.*`` — the pipeline-native online-mix compiler
- ``streams.*`` — the dload↔tdseries bridge
- ``frame_datasets.*`` — torch Dataset adapters
"""
