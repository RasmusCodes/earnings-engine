"""fundalpha - point-in-time earnings prediction and fundamental alpha research engine.

Layers
- http / edgar / xbrl : SEC EDGAR acquisition with an immutable raw cache and every fact's filing timestamp
- pit                 : point-in-time (as-of) quarterly snapshots built from all reported vintages
- events / guidance   : earnings-release timestamps (8-K Item 2.02) and management outlook extracted from releases
- features / models   : forecast-origin features, benchmarks and ML models
- walkforward / evaluate : strict temporal validation and statistics
"""
__version__ = "0.1.0"
