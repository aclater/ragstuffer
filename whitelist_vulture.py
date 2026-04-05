# Vulture whitelist — false positives and intentional unused code.
# Run: vulture . whitelist_vulture.py --min-confidence 80 --exclude tests,test_*.py,.venv
#
# Findings at min-confidence 80 (2026-04-04):
#   common.py:108 — unused variable 'attrs' in HTMLParser.handle_starttag (required by API)
#   ragstuffer.py:701 — unused 'args'/'fmt' in logging formatter (required by logging API)
#   ragstuffer.py:729 — unused 'frame' in signal handler (required by signal API)

# HTMLParser API requires attrs parameter
attrs = None  # common.py:108

# logging.Formatter and signal handler signatures require these
args = None  # ragstuffer.py:701
fmt = None  # ragstuffer.py:701
frame = None  # ragstuffer.py:729
