"""Pull a single hourly recording out of the Sierra Nevada ELF Zenodo ZIP
without downloading the whole ~27 GB archive (HTTP range requests).

Usage:
    python pull_one_hour.py --out <dir> [--year 2014] [--member <zip-path>]
    python pull_one_hour.py --list 20            # list first 20 sensor files

Requires: pip install remotezip
"""

from __future__ import annotations

import argparse
import os

from remotezip import RemoteZip

YEAR_URLS = {
    "2013": "https://zenodo.org/records/6348930/files/2013.zip?download=1",
    "2014": "https://zenodo.org/records/6348691/files/2014.zip?download=1",
    "2015": "https://zenodo.org/records/6348773/files/2015.zip?download=1",
    "2016": "https://zenodo.org/records/6348838/files/2016.zip?download=1",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2014", choices=sorted(YEAR_URLS))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sierra_nevada_elf"))
    ap.add_argument("--member", default=None,
                    help="exact zip member path; default = first sensor_1 file")
    ap.add_argument("--list", type=int, default=0,
                    help="just list the first N sensor data members and exit")
    args = ap.parse_args()

    url = YEAR_URLS[args.year]
    with RemoteZip(url) as z:
        names = z.namelist()
        sensors = [n for n in names
                   if "sensor" in n.lower() and not n.lower().endswith(".txt")]
        if args.list:
            for n in sensors[:args.list]:
                print(n)
            print(f"... {len(sensors)} sensor files total")
            return

        member = args.member or sensors[0]
        info_member = member + "_info.txt"
        os.makedirs(args.out, exist_ok=True)

        z.extract(member, path=args.out)
        data_path = os.path.join(args.out, member.replace("/", os.sep))
        try:
            with z.open(info_member) as f:
                info = f.read()
            with open(data_path + "_info.txt", "wb") as g:
                g.write(info)
        except KeyError:
            print("(no companion _info.txt found; 256 Hz will be assumed)")

        print("data file :", data_path, os.path.getsize(data_path), "bytes")
        print("info file :", data_path + "_info.txt")


if __name__ == "__main__":
    main()
