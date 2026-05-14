"""Registry of known GW events for the catcher's real-event mode.

GPS times sourced from GWOSC (https://gwosc.org). Add more entries as
needed; the registry is just a convenience map used by the
end-to-end `run_event_catcher` runner.
"""

from __future__ import annotations

KNOWN_EVENTS: dict[str, dict] = {
    "GW150914": {"gps": 1126259462.4, "detectors": ("H1", "L1")},
    "GW170817": {"gps": 1187008882.4, "detectors": ("H1", "L1", "V1")},
    "GW170814": {"gps": 1186741861.5, "detectors": ("H1", "L1", "V1")},
}
